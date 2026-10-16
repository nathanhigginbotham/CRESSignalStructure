"""
BatchGenScattering.py

Generate CRES events with gas scattering, using the same trap, LO,
particles and phases as BatchGenFast.py.

Event i of file f draws its electron from default_rng([particle_seed, f, i])
and its initial phases from default_rng([phase_seed, f, i]), exactly as
BatchGenFast.py does. With the seeds of an existing BatchGenFast run, event
i here is therefore the same electron as event i of that run's file f,
started with the same phases, so the scattered and unscattered datasets can
be compared event by event. Each event stores its stream index as
'Event index'.

The scatter draws (time to scatter, energy loss, deflection) come from
SeedSequence([scatter_seed, f]), so a run is reproducible. Before the first
scatter the waveform follows the same physics as BatchGenFast.py (field-line
orbit average, physical chirp, arange time axis, the same oversampling and
anti-alias filter); after a scatter the carrier and sidebands step to the
new energy and pitch.

Events are variable length: a scatter that leaves the pitch below the
trapping threshold (86.37 deg on axis at 4 mT) ejects the electron and
ends the event early.

The gas is an 80/20 atomic tritium / helium-4 mixture at --density-cm3.

Usage example (the electrons of random_phase_v2 test file 000):
    python BatchGenScattering.py --events 1500 --prefix scattered_v2 \\
        --particle-seed 2020 --phase-seed 1020 --file-index 0 \\
        --workers 8 --outdir data/scattered_v2
"""
import argparse
import json
import logging
import os
import time

import numpy as np

from CRESSignalStructure.CircularWaveguide import CircularWaveguide
from CRESSignalStructure.scattering import (
    InelasticCrossSection, ElasticCrossSection, GasModel)
from CRESSignalStructure.EnsembleGenerator import generate_scattering_ensemble

from BatchGenFast import (BACKGROUND_FIELD_T, COIL_RADIUS_M, DEFAULT_TRAP_DEPTH_T,
                          SAMPLE_RATE_HZ, WAVEGUIDE_RADIUS_M, _git_commit,
                          build_trap, check_carrier_in_band, choose_lo,
                          coil_current, frequency_plan, make_particle_generator,
                          make_phase_generator)


def build_gas(density_cm3):
    """80/20 atomic-tritium / helium-4 mixture at the given total density."""
    n_total = density_cm3 * 1e6          # cm^-3 -> m^-3
    n_T = 0.8 * n_total
    n_He = 0.2 * n_total
    # Atomic tritium: inelastic uses "H" params (identical electron structure),
    # elastic uses Z=1, A=3.016. Helium-4: Z=2, A=4.003.
    return GasModel([
        (InelasticCrossSection("H"), n_T),
        (ElasticCrossSection(Z=1, A=3.016), n_T),
        (InelasticCrossSection("He"), n_He),
        (ElasticCrossSection(Z=2, A=4.003), n_He),
    ])


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate CRES events with gas scattering")
    p.add_argument("--events", type=int, default=1500,
                   help="Events, taken as indices 0..N-1 of the stream of --file-index")
    p.add_argument("--file-index", type=int, default=0,
                   help="Seed stream (file index of the matching BatchGenFast run)")
    p.add_argument("--outdir", default="data/scattered_run")
    p.add_argument("--prefix", default="run")
    p.add_argument("--particle-seed", type=int, required=True)
    p.add_argument("--phase-seed", type=int, required=True)
    p.add_argument("--scatter-seed", type=int, default=0,
                   help="Seed for the scatter draws")

    # Particle box, LO and trap: BatchGenFast.py defaults
    p.add_argument("--energy-min", type=float, default=18500.0)
    p.add_argument("--energy-max", type=float, default=18600.0)
    p.add_argument("--pitch-min", type=float, default=86.5, help="Degrees")
    p.add_argument("--pitch-max", type=float, default=90.0, help="Degrees")
    p.add_argument("--r-max", type=float, default=5e-3, help="Maximum start radius (m)")
    p.add_argument("--max-order", type=int, default=8)
    p.add_argument("--trap-depth", type=float, default=DEFAULT_TRAP_DEPTH_T)
    p.add_argument("--lo-mode", choices=("quarter", "centre"), default="quarter")
    p.add_argument("--lo-freq", type=float, default=None,
                   help="Explicit LO (Hz), overriding --lo-mode")

    # Gas and event length: scattered_v1 values
    p.add_argument("--density-cm3", type=float, default=1e12,
                   help="Total gas number density (cm^-3)")
    p.add_argument("--max-event-time", type=float, default=500e-6, help="Seconds")

    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes (1 runs in the main process)")
    p.add_argument("--fft", action="store_true", help="Also write FFT power files")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # The simulator logs every scatter at INFO; keep the progress lines only
    logging.getLogger("CRESSignalStructure.scattering").setLevel(logging.ERROR)
    os.makedirs(args.outdir, exist_ok=True)

    try:
        trap = build_trap(args.trap_depth)
    except ValueError as exc:
        p.error(str(exc))
    wg = CircularWaveguide(radius=WAVEGUIDE_RADIUS_M)
    ranges_deg = {'energy': (args.energy_min, args.energy_max),
                  'pitch': (args.pitch_min, args.pitch_max), 'r': (0.0, args.r_max)}
    try:
        plan = frequency_plan(trap, wg, args.trap_depth, ranges_deg)
        lo_freq = choose_lo(trap, wg, ranges_deg, plan, SAMPLE_RATE_HZ,
                            args.lo_mode, args.lo_freq)
        carrier_dm = check_carrier_in_band(plan, lo_freq, SAMPLE_RATE_HZ)
    except ValueError as exc:
        p.error(str(exc))
    ranges = {'energy': (args.energy_min, args.energy_max),
              'pitch': (np.radians(args.pitch_min), np.radians(args.pitch_max)),
              'r': (0.0, args.r_max)}
    sim_config = {'sample_rate': SAMPLE_RATE_HZ, 'lo_freq': lo_freq,
                  'max_event_time': args.max_event_time, 'max_order': args.max_order}

    config = {**vars(args), 'lo_freq': lo_freq, 'generator': 'BatchGenScattering.py',
              'css_commit': _git_commit(),
              'scatter_seed_sequence': [args.scatter_seed, args.file_index],
              'gas': '80/20 atomic T / He-4; inelastic H and He, elastic Z=1 A=3.016 '
                     'and Z=2 A=4.003',
              'physics': {'trap_depth_T': args.trap_depth,
                          'background_field_T': BACKGROUND_FIELD_T,
                          'coil_radius_m': COIL_RADIUS_M,
                          'coil_current_A': coil_current(args.trap_depth),
                          'waveguide_radius_m': WAVEGUIDE_RADIUS_M,
                          'sample_rate_Hz': SAMPLE_RATE_HZ,
                          'min_trapped_pitch_deg': plan['min_trapped_pitch_deg'],
                          'lo_mode': 'explicit' if args.lo_freq is not None else args.lo_mode,
                          'downmixed_carrier_range_Hz': list(carrier_dm),
                          'downmixed_label': 'signed, f - LO',
                          'chirp': 'physical (0.5 k t^2, gamma^3), carried across scatters',
                          'time_axis': 'arange / fs'}}
    file_base = f"{args.prefix}_{args.file_index:03d}"
    with open(os.path.join(args.outdir, f"run_config_{file_base}.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"LO {lo_freq / 1e9:.6f} GHz, carrier {carrier_dm[0] / 1e6:.1f} to "
          f"{carrier_dm[1] / 1e6:.1f} MHz, gas {args.density_cm3:.2e} cm^-3, window "
          f"{args.max_event_time * 1e6:.0f} us, {args.events} events of stream "
          f"{args.file_index}, commit {config['css_commit'][:10]}")

    final = os.path.join(args.outdir, f"{file_base}_signal.h5")
    if os.path.exists(final):
        print(f"{final} exists, nothing to do")
        return 0
    partial = final + ".partial"
    fft_path = os.path.join(args.outdir, f"{file_base}_fft.h5") if args.fft else None

    t0 = time.time()
    generate_scattering_ensemble(
        output_file=partial,
        n_events=args.events,
        particle_generator=make_particle_generator(ranges, args.particle_seed,
                                                   args.file_index),
        trap=trap,
        waveguide=wg,
        gas_model=build_gas(args.density_cm3),
        sim_config=sim_config,
        fft_output_file=fft_path,
        use_multiprocessing=args.workers > 1,
        verbose=True,
        seed=[args.scatter_seed, args.file_index],
        phase_generator=make_phase_generator(args.phase_seed, args.file_index),
        processes=args.workers,
    )
    os.replace(partial, final)
    dt = time.time() - t0
    print(f"wrote {final}: {args.events} events in {dt:.0f} s "
          f"({args.events / dt:.2f} events/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
