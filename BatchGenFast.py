"""
BatchGenFast.py

Generate CRES datasets with the batched generator (BatchedSignalGenerator):
CPU workers for the per-event physics, float64 synthesis on CPU or CUDA.
Output files have the same layout as BatchGen.py plus the observable truth
labels ('Mean cyclotron frequency', 'Mean downmixed cyclotron frequency',
'Axial frequency') and the initial phases.

Reproducibility: event i of file f draws its particle from
default_rng([particle_seed, f, i]) and its phases from
default_rng([phase_seed, f, i]). Every file is therefore an independent,
reproducible stream (unlike BatchGen.py --seed, which repeats one stream in
every file), and files can be split across jobs with --first-file.

Each file is written to <name>.partial and renamed when complete, so an
interrupted run resumes by skipping finished files.

Sparse output (--sparse): instead of the dense records, write each file
index as a directory <prefix>_<fff>_sparse holding the noiseless spectrum
X / fft_sigma(N, T_ref) of every event, keeping only the bins whose
amplitude at --sparse-tmin exceeds --sparse-rel of the per-component noise
std (about 45 kB per 40 us event instead of 320 kB), plus labels, SNRs and
the full truth table. The layout is that of the detection repository's
full-band cache (CRESSignalStructure/SparseSpectrum.py), so a directory is
read there directly. Event i of the sparse unit f is the same electron as
event i of dense file f (same seeds, draws and drop logic); --also-dense
writes both for validation (scripts/check_sparse_generation.py).

Trap depth (--trap-depth, tesla, default 4 mT): the HarmonicField coil
current is set so that the coil field at the trap centre equals the depth,
which therefore sets the field map, the LO and the trapping threshold. On
axis an electron is trapped when sin^2(pitch) > (B0 - depth) / B0 (86.37 deg
at 4 mT, 79.70 deg at 32 mT; off axis the threshold is lower). A pitch range
reaching below the on-axis threshold is refused before anything is
generated, as is an LO that would put the carrier itself outside the
digitiser band; within a valid range, the only events dropped are the few
whose physics fails numerically (pitch within ~2e-3 deg of 90).

LO (--lo-mode): 'quarter' (default, unchanged) puts the reference electron
(mid energy, mid pitch, on axis) at +fs/4; 'centre' centres the carrier
range of the sampled box in the band, needed when that range is wider than
fs/2 (32 mT, pitch 80-90: 828 MHz); --lo-freq sets it explicitly. Downmixed
frequency labels are signed (f - LO), so a carrier below the LO is negative.

Usage example (the random_phase_v1 test layout):
    python BatchGenFast.py --files 10 --events 10000 --prefix test_100K \
        --pitch-min 86.5 --pitch-max 90 --energy-min 18500 --energy-max 18600 \
        --phase-seed 1020 --particle-seed 2020 --device cuda --workers 8 \
        --outdir data/random_phase_v2/test
"""
import argparse
import json
import logging
import os
import subprocess
import time

import numpy as np
import scipy.constants as sc

from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.Particle import Particle
from CRESSignalStructure.SpectrumCalculator import SpectrumCalculator
from CRESSignalStructure.BatchedSignalGenerator import generate_ensemble_batched
from CRESSignalStructure.SparseSpectrum import SparseSpectrumWriter


def _git_commit() -> str:
    """Commit of the code that generated the data, for provenance."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


# Fixed geometry of BatchGen.py; only the trap depth is a run option
DEFAULT_TRAP_DEPTH_T = 4e-3
BACKGROUND_FIELD_T = 1.0
COIL_RADIUS_M = 1.25e-2
WAVEGUIDE_RADIUS_M = 5e-3
SAMPLE_RATE_HZ = 1e9
# Pitch at which the frequency plan evaluates the top of the range: the
# physics fails numerically within ~2e-3 deg of 90 (those events are dropped)
PLAN_PITCH_MAX_DEG = 89.99


def coil_current(trap_depth: float, r_coil: float = COIL_RADIUS_M) -> float:
    """Coil current (A) whose central field mu0 I / (2 R) equals the trap depth."""
    return 2 * trap_depth * r_coil / sc.mu_0


def build_trap(trap_depth: float, field_cls=HarmonicField):
    """HarmonicField of the BatchGen.py geometry with the given depth (T).

    field_cls lets the equivalence benchmark build the same trap from a
    reference copy of the package."""
    if not 0.0 < trap_depth < BACKGROUND_FIELD_T:
        raise ValueError(f"trap depth must be in (0, {BACKGROUND_FIELD_T}) T, got {trap_depth}")
    return field_cls(radius=COIL_RADIUS_M, current=coil_current(trap_depth),
                     background=BACKGROUND_FIELD_T)


def min_trapped_pitch_deg(trap_depth: float, b0: float = BACKGROUND_FIELD_T) -> float:
    """On-axis trapping threshold (deg): sin^2(theta) = B_min / B_max = (B0 - d) / B0.

    The coil field opposes the background, so |B| dips by the depth at the
    centre and recovers B0 far from the coil. Off axis the dip is deeper and
    the threshold lower, so the on-axis value bounds every start radius."""
    return float(np.degrees(np.arcsin(np.sqrt((b0 - trap_depth) / b0))))


def _tones(trap, wg, ke, pitch_deg, r):
    """(mean carrier, axial frequency) in Hz of one electron, as the generator computes them."""
    p = Particle(ke=ke, startPos=np.array([r, 0.0, 0.0]), pitchAngle=np.radians(pitch_deg))
    f = SpectrumCalculator(trap, wg, p).get_peak_frequency(np.array([0, 1]))
    return float(f[0]), float(f[1] - f[0])


def frequency_plan(trap, wg, trap_depth, ranges_deg):
    """Check the sampled box is trapped and return its carrier and f_a extremes.

    ranges_deg: {'energy': (eV, eV), 'pitch': (deg, deg), 'r': (m, m)}. The
    carrier falls with pitch, radius and energy, and f_a rises with pitch
    and radius, so the extremes are at the corners of the box. Raises
    ValueError if the pitch range reaches untrapped values or the physics
    fails at a corner, so a bad range stops the run before any event is
    generated rather than dropping events one by one."""
    (e_lo, e_hi), (p_lo, p_hi), (r_lo, r_hi) = (ranges_deg[k] for k in ('energy', 'pitch', 'r'))
    p_min_trap = min_trapped_pitch_deg(trap_depth)
    if p_lo <= p_min_trap:
        raise ValueError(
            f"pitch range starts at {p_lo} deg but a {trap_depth * 1e3:g} mT trap only "
            f"traps pitches above {p_min_trap:.4f} deg on axis; raise --pitch-min or "
            f"--trap-depth (depth needed: {BACKGROUND_FIELD_T * np.cos(np.radians(p_lo))**2:.4g} T)")
    corners = {}
    for ke in (e_lo, e_hi):
        for pitch in (p_lo, min(p_hi, PLAN_PITCH_MAX_DEG)):
            for r in (r_lo, r_hi):
                try:
                    corners[(ke, pitch, r)] = _tones(trap, wg, ke, pitch, r)
                except Exception as exc:
                    raise ValueError(f"physics fails at energy {ke} eV, pitch {pitch} deg, "
                                     f"radius {r} m ({exc!r})") from exc
    fc = [v[0] for v in corners.values()]
    fa = [v[1] for v in corners.values()]
    return {'min_trapped_pitch_deg': p_min_trap,
            'carrier_Hz': (min(fc), max(fc)), 'axial_Hz': (min(fa), max(fa))}


def choose_lo(trap, wg, ranges_deg, plan, sample_rate, mode, lo_freq=None):
    """LO frequency (Hz): explicit, 'quarter' (reference electron at +fs/4,
    the BatchGen.py convention) or 'centre' (carrier range centred)."""
    if lo_freq is not None:
        return float(lo_freq)
    if mode == 'quarter':
        (e_lo, e_hi), (p_lo, p_hi) = ranges_deg['energy'], ranges_deg['pitch']
        ref = Particle(ke=(e_lo + e_hi) / 2, startPos=np.zeros(3),
                       pitchAngle=np.radians((p_lo + p_hi) / 2))
        return SpectrumCalculator(trap, wg, ref).get_peak_frequency(0) - sample_rate / 4
    if mode == 'centre':
        return 0.5 * (plan['carrier_Hz'][0] + plan['carrier_Hz'][1])
    raise ValueError(f"unknown LO mode {mode!r}")


def check_carrier_in_band(plan, lo_freq, sample_rate):
    """Raise if the carrier itself would alias; warn if it reaches the filter roll-off.

    The anti-alias Butterworth (order 8, zero-phase) has its -6 dB point at
    fs/2 and passes 0.5 dB less at about 0.42 fs."""
    lo_c, hi_c = (f - lo_freq for f in plan['carrier_Hz'])
    edge = max(abs(lo_c), abs(hi_c))
    if edge >= sample_rate / 2:
        raise ValueError(
            f"downmixed carrier spans {lo_c / 1e6:.1f} to {hi_c / 1e6:.1f} MHz, outside the "
            f"+-{sample_rate / 2e6:.0f} MHz band, so it would alias; use --lo-mode centre "
            f"(range width {(hi_c - lo_c) / 1e6:.1f} MHz) or set --lo-freq")
    if edge > 0.45 * sample_rate:
        logging.warning("downmixed carrier reaches %.1f MHz, in the anti-alias filter roll-off",
                        edge / 1e6)
    return lo_c, hi_c


def make_particle_generator(ranges, particle_seed, file_idx):
    """Particle i of a file drawn from its own (seed, file, i) stream.

    The draw order (energy, pitch, z, radius, azimuth) is that of the
    io-pipeline BatchGen.py that produced random_phase_v1, so the same seeds
    give the same particles and the two datasets can be compared event by
    event."""
    def gen(i):
        rng = np.random.default_rng([particle_seed, file_idx, i])
        ke = rng.uniform(*ranges['energy'])
        pitch = rng.uniform(*ranges['pitch'])
        rng.uniform(0.0, 0.0)       # z draw (always 0); keeps the stream aligned
        r_min, r_max = ranges['r']
        r = np.sqrt((r_max**2 - r_min**2) * rng.uniform(0, 1) + r_min**2)
        theta = rng.uniform(0.0, 2 * np.pi)
        p = Particle(ke=ke, startPos=np.array([r * np.cos(theta), r * np.sin(theta), 0.0]),
                     pitchAngle=pitch)
        # The raw draws, for the sparse truth table (r and theta are not
        # recoverable bitwise from the start position)
        p.draw_values = {'energy_ev': ke, 'pitch_rad': pitch, 'z_m': 0.0,
                         'radius_m': r, 'azimuth_rad': theta}
        return p
    return gen


def make_phase_generator(phase_seed, file_idx):
    """Initial (phi_c, phi_a) of event i drawn from its own stream."""
    def gen(i):
        rng = np.random.default_rng([phase_seed, file_idx, i])
        return float(rng.uniform(0.0, 2 * np.pi)), float(rng.uniform(0.0, 2 * np.pi))
    return gen


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate CRES batches with the batched generator")
    p.add_argument("--files", type=int, default=1, help="Number of files")
    p.add_argument("--first-file", type=int, default=0,
                   help="Index of the first file (to split a run across jobs)")
    p.add_argument("--events", type=int, default=10000, help="Events per file")
    p.add_argument("--outdir", default="data/batch_run")
    p.add_argument("--prefix", default="run")
    p.add_argument("--energy-min", type=float, default=18500.0)
    p.add_argument("--energy-max", type=float, default=18600.0)
    p.add_argument("--pitch-min", type=float, default=86.5, help="Degrees")
    p.add_argument("--pitch-max", type=float, default=90.0, help="Degrees")
    p.add_argument("--r-max", type=float, default=5e-3, help="Maximum start radius (m)")
    p.add_argument("--acq-time", type=float, default=40e-6, help="Seconds")
    p.add_argument("--max-order", type=int, default=8)
    p.add_argument("--trap-depth", type=float, default=DEFAULT_TRAP_DEPTH_T,
                   help="Trap depth (T): coil field at the trap centre, against the 1 T background")
    p.add_argument("--lo-mode", choices=("quarter", "centre"), default="quarter",
                   help="'quarter': reference electron at +fs/4 (default); "
                        "'centre': carrier range centred in the band")
    p.add_argument("--lo-freq", type=float, default=None,
                   help="Explicit LO (Hz), overriding --lo-mode")
    p.add_argument("--phase-seed", type=int, required=True)
    p.add_argument("--particle-seed", type=int, required=True)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=1, help="Processes for the physics stage")
    p.add_argument("--fft", action="store_true", help="Also write FFT power files")
    p.add_argument("--sparse", action="store_true",
                   help="Write sparse noiseless spectra (full-band cache layout) instead of dense records")
    p.add_argument("--also-dense", action="store_true",
                   help="With --sparse, also write the dense records (validation)")
    p.add_argument("--sparse-tmin", type=float, default=2.0,
                   help="Coldest training temperature (K) the threshold refers to")
    p.add_argument("--sparse-rel", type=float, default=1e-2,
                   help="Keep bins above this fraction of the per-component noise std at --sparse-tmin")
    p.add_argument("--sparse-tref", type=float, default=10.0,
                   help="Reference temperature (K) of the stored normalised spectrum")
    args = p.parse_args(argv)
    if args.also_dense and not args.sparse:
        p.error("--also-dense needs --sparse")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(args.outdir, exist_ok=True)

    # Physics, as in BatchGen.py, with the trap depth as an option
    trap_depth, b0, r_coil = args.trap_depth, BACKGROUND_FIELD_T, COIL_RADIUS_M
    try:
        trap = build_trap(trap_depth)
    except ValueError as exc:
        p.error(str(exc))
    wg = CircularWaveguide(radius=WAVEGUIDE_RADIUS_M)
    f_digitiser = SAMPLE_RATE_HZ
    ranges_deg = {'energy': (args.energy_min, args.energy_max),
                  'pitch': (args.pitch_min, args.pitch_max), 'r': (0.0, args.r_max)}
    # Fail before anything is written if the box is not trapped or the
    # carrier would alias
    try:
        plan = frequency_plan(trap, wg, trap_depth, ranges_deg)
        lo_freq = choose_lo(trap, wg, ranges_deg, plan, f_digitiser, args.lo_mode, args.lo_freq)
        carrier_dm = check_carrier_in_band(plan, lo_freq, f_digitiser)
    except ValueError as exc:
        p.error(str(exc))
    sim_config = {'sample_rate': f_digitiser, 'lo_freq': lo_freq,
                  'acq_time': args.acq_time, 'max_order': args.max_order}
    ranges = {'energy': (args.energy_min, args.energy_max),
              'pitch': (np.radians(args.pitch_min), np.radians(args.pitch_max)),
              'r': (0.0, args.r_max)}

    # Every physical constant goes in the run config so a dataset can be
    # checked against its intended geometry without reading the code.
    config = {**vars(args), 'lo_freq': lo_freq, 'generator': 'BatchGenFast.py',
              'css_commit': _git_commit(),
              'physics': {'trap_depth_T': trap_depth, 'background_field_T': b0,
                          'coil_radius_m': r_coil,
                          'coil_current_A': coil_current(trap_depth, r_coil),
                          'waveguide_radius_m': WAVEGUIDE_RADIUS_M, 'sample_rate_Hz': f_digitiser,
                          'min_trapped_pitch_deg': plan['min_trapped_pitch_deg'],
                          'lo_mode': 'explicit' if args.lo_freq is not None else args.lo_mode,
                          'carrier_range_Hz': list(plan['carrier_Hz']),
                          'downmixed_carrier_range_Hz': list(carrier_dm),
                          'axial_range_Hz': list(plan['axial_Hz']),
                          'downmixed_label': 'signed, f - LO',
                          'chirp': 'physical (0.5 k t^2, gamma^3)',
                          'time_axis': 'arange / fs'}}
    cfg_path = os.path.join(args.outdir, f"run_config_{args.prefix}_{args.first_file:03d}.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"trap depth {trap_depth * 1e3:g} mT (pitch > {plan['min_trapped_pitch_deg']:.3f} deg "
          f"trapped), LO {lo_freq / 1e9:.6f} GHz ({config['physics']['lo_mode']}), carrier "
          f"{carrier_dm[0] / 1e6:.1f} to {carrier_dm[1] / 1e6:.1f} MHz, f_a "
          f"{plan['axial_Hz'][0] / 1e6:.1f} to {plan['axial_Hz'][1] / 1e6:.1f} MHz, "
          f"commit {config['css_commit'][:10]}, device {args.device}")

    t_start = time.time()
    total, dropped = 0, 0
    for fi in range(args.first_file, args.first_file + args.files):
        final = os.path.join(args.outdir, f"{args.prefix}_{fi:03d}_signal.h5")
        sparse_final = os.path.join(args.outdir, f"{args.prefix}_{fi:03d}_sparse")
        # A unit is done when every requested output exists; only the
        # missing ones are regenerated (same seeds, so they still match)
        want_dense = not args.sparse or args.also_dense
        need_dense = want_dense and not os.path.exists(final)
        need_sparse = args.sparse and not os.path.exists(sparse_final)
        if not (need_dense or need_sparse):
            print(f"[file {fi}] exists, skipping")
            continue
        partial = final + ".partial"
        fft_final = (os.path.join(args.outdir, f"{args.prefix}_{fi:03d}_fft.h5")
                     if args.fft and need_dense else None)
        sparse_writer = None
        if need_sparse:
            sparse_writer = SparseSpectrumWriter(
                sparse_final + ".partial", source_file=os.path.abspath(final),
                device=args.device, system_temp_ref=args.sparse_tref,
                temp_min=args.sparse_tmin, rel_threshold=args.sparse_rel,
                provenance={**config, 'file_index': fi})
        t0 = time.time()
        n_drop = generate_ensemble_batched(
            output_file=partial if need_dense else None, n_events=args.events,
            particle_generator=make_particle_generator(ranges, args.particle_seed, fi),
            phase_generator=make_phase_generator(args.phase_seed, fi),
            trap=trap, waveguide=wg, sim_config=sim_config,
            fft_output_file=(fft_final + ".partial") if fft_final else None,
            device=args.device, batch_size=args.batch_size, n_workers=args.workers,
            sparse_writer=sparse_writer)
        if need_dense:
            os.replace(partial, final)
        if fft_final:
            os.replace(fft_final + ".partial", fft_final)
        if need_sparse:
            os.replace(sparse_final + ".partial", sparse_final)
        dt = time.time() - t0
        total += args.events - n_drop
        dropped += n_drop
        print(f"[file {fi}] {args.events - n_drop} events ({n_drop} dropped) in {dt:.0f} s, "
              f"{(args.events - n_drop) / dt:.1f} ev/s", flush=True)
    elapsed = time.time() - t_start
    print(f"DONE {total} events, {dropped} dropped, {elapsed:.0f} s"
          + (f", {total / elapsed:.1f} ev/s" if total else ""))


if __name__ == "__main__":
    main()
