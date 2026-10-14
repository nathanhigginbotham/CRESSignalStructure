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

    # Physics, as in BatchGen.py
    trap_depth, b0, r_coil = 4e-3, 1.0, 1.25e-2
    trap = HarmonicField(radius=r_coil, current=2 * trap_depth * r_coil / sc.mu_0, background=b0)
    wg = CircularWaveguide(radius=5e-3)
    f_digitiser = 1e9
    ref = Particle(ke=(args.energy_min + args.energy_max) / 2, startPos=np.zeros(3),
                   pitchAngle=np.radians((args.pitch_min + args.pitch_max) / 2))
    lo_freq = SpectrumCalculator(trap, wg, ref).get_peak_frequency(0) - f_digitiser / 4
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
                          'coil_current_A': 2 * trap_depth * r_coil / sc.mu_0,
                          'waveguide_radius_m': 5e-3, 'sample_rate_Hz': f_digitiser,
                          'chirp': 'physical (0.5 k t^2, gamma^3)',
                          'time_axis': 'arange / fs'}}
    cfg_path = os.path.join(args.outdir, f"run_config_{args.prefix}_{args.first_file:03d}.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"LO {lo_freq / 1e9:.6f} GHz, commit {config['css_commit'][:10]}, device {args.device}")

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
