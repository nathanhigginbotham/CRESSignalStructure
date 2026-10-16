"""
check_scattering_generation.py

Check that BatchGenScattering.py produces BatchGenFast.py's events when
the gas does nothing, and the same waveform up to the first scatter when
it does.

1. LO: the scattering run's LO equals the one BatchGenFast.py chooses for
   the same box (or --expect-lo, e.g. from a run config).
2. No scatter: with the gas density at 1e-30 cm^-3 and a 40 us window,
   each event must match SignalGenerator (the reference that BatchGenFast's
   equivalence gate checks against) for the same electron and phases.
3. With scatter: at the production density, the waveform before the first
   scatter must match the unscattered one, away from the anti-alias filter
   transient at the scatter itself.

Usage:
    python scripts/check_scattering_generation.py --particle-seed 2020 \\
        --phase-seed 1020 --file-index 0 --events 12 \\
        --expect-lo 26667465723.059723
"""
import argparse
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CRESSignalStructure.CircularWaveguide import CircularWaveguide  # noqa: E402
from CRESSignalStructure.SignalGenerator import SignalGenerator  # noqa: E402
from CRESSignalStructure.SpectrumCalculator import SpectrumCalculator  # noqa: E402
import BatchGenScattering  # noqa: E402
from BatchGenFast import (WAVEGUIDE_RADIUS_M, build_trap,  # noqa: E402
                          make_particle_generator)

FS = 1e9
# Samples to skip either side of a scatter: the zero-phase Butterworth
# (order 8) spreads the step over a few hundred samples at 1 GHz
GUARD = 500


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def read(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k, d in f["Data"].items():
            a = d.attrs
            out[int(a["Event index"])] = (d[:], dict(a))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particle-seed", type=int, required=True)
    p.add_argument("--phase-seed", type=int, required=True)
    p.add_argument("--file-index", type=int, default=0)
    p.add_argument("--events", type=int, default=12)
    p.add_argument("--expect-lo", type=float, default=None)
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args()
    common = ["--particle-seed", str(args.particle_seed), "--phase-seed",
              str(args.phase_seed), "--file-index", str(args.file_index),
              "--events", str(args.events)]
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        BatchGenScattering.main(common + ["--outdir", f"{tmp}/off", "--density-cm3", "1e-30",
                                          "--max-event-time", "40e-6"])
        BatchGenScattering.main(common + ["--outdir", f"{tmp}/on"])
        off = read(f"{tmp}/off/run_{args.file_index:03d}_signal.h5")
        on = read(f"{tmp}/on/run_{args.file_index:03d}_signal.h5")

    trap = build_trap(4e-3)
    wg = CircularWaveguide(radius=WAVEGUIDE_RADIUS_M)
    lo = next(iter(off.values()))[1]["LO frequency [Hertz]"]
    print(f"== LO {lo:.6f} Hz")
    if args.expect_lo is not None:
        good = lo == args.expect_lo
        ok &= good
        print(f"   expected {args.expect_lo:.6f} Hz: {'match' if good else 'MISMATCH'}")

    ranges = {"energy": (18500.0, 18600.0), "pitch": (np.radians(86.5), np.radians(90.0)),
              "r": (0.0, 5e-3)}
    particles = make_particle_generator(ranges, args.particle_seed, args.file_index)
    print("== no scatter against SignalGenerator (40 us)")
    worst = 0.0
    for i, (s, a) in sorted(off.items()):
        assert a["n_scatters"] == 0, f"event {i} scattered at 1e-30 cm^-3"
        ref = SignalGenerator(SpectrumCalculator(trap, wg, particles(i)), FS, lo, 40e-6)
        r = ref.generate_signal(8, a["Cyclotron phase [rad]"], a["Axial phase [rad]"])[1]
        n = min(len(r), len(s))
        e = rel_l2(s[:n], r[:n])
        worst = max(worst, e)
        print(f"   event {i:3d} pitch {a['Pitch angle [degrees]']:.3f} radius "
              f"{np.hypot(*a['Starting position [metres]'][:2]) * 1e3:.2f} mm  "
              f"length {len(s)} vs {len(r)}  relL2 {e:.1e}")
    good = worst < args.tol
    ok &= good
    print(f"   worst relL2 {worst:.1e} (tolerance {args.tol:g}): {'pass' if good else 'FAIL'}")

    print("== production density: waveform before the first scatter")
    n_checked = 0
    for i, (s, a) in sorted(on.items()):
        t = a.get("scatter_times [seconds]")
        if a["n_scatters"] == 0 or t is None:
            continue
        end = min(int(t[0] * FS), len(off[i][0])) - GUARD
        if end < GUARD:
            continue
        e = rel_l2(s[:end], off[i][0][:end])
        n_checked += 1
        good = e < 1e-3
        ok &= good
        print(f"   event {i:3d} first scatter at {t[0] * 1e6:6.2f} us, "
              f"{a['n_scatters']} scatters, escaped {bool(a['escaped'])}: "
              f"relL2 before it {e:.1e} {'pass' if good else 'FAIL'}")
    lengths = [len(s) for s, _ in on.values()]
    print(f"   {n_checked} events checked; lengths {min(lengths)} to {max(lengths)} samples; "
          f"{sum(a['n_scatters'] > 0 for _, a in on.values())} of {len(on)} scattered")
    print("ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
