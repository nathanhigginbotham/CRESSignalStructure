"""
benchmark_batched_generation.py

Throughput and equivalence benchmark for CRES event generation.

Three paths are timed on the same particles and phases, with the physics
setup of BatchGen.py (HarmonicField R = 1.25 cm, B0 = 1 T, 5 mm circular
waveguide, 1 GHz sampling, max order 8) and the trap depth, pitch range and
LO convention of BatchGenFast.py (defaults: 4 mT, pitch 86.5-89.9 deg,
LO = f_c(ref) - 250 MHz; --trap-depth 32e-3 --pitch-min 80 --lo-mode centre
gates the broad-range geometry; --trap bathtub --coil-z --background gate the
two-coil trap, and --max-order the sideband order of the run):

  original  SignalGenerator.generate_signal from a reference git commit
            (default 3cf55ca, before the physics caching), loaded from
            `git archive` into a temporary package
  cached    SignalGenerator.generate_signal from the working tree
  batched   BatchedSignalGenerator (physics on CPU, synthesis on --device)

For each acquisition time it reports events/s and the physics and synthesis
cost per event, and checks equivalence: cached against original (expected
bitwise) and batched against cached (relative L2 per event, target <= 1e-10).

Usage
-----
    PYTHONPATH=. python scripts/benchmark_batched_generation.py --device cpu
    PYTHONPATH=. python scripts/benchmark_batched_generation.py --device cuda \\
        --batch-sizes 16 64 256 --n-events 512 --n-workers 4

Timings on a shared or loaded machine are noisy; compare paths from the same
run only.
"""

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.constants as sc

TOL = 1e-10


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu", help="torch device for synthesis (cpu, cuda)")
    ap.add_argument("--acq-times", type=float, nargs="+", default=[40e-6, 200e-6])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--n-events", type=int, default=16,
                    help="events per batched timing (and particle pool size)")
    ap.add_argument("--n-ref-events", type=int, default=4,
                    help="events timed on the single-process original and cached paths")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="processes for the batched physics stage")
    ap.add_argument("--phasors", nargs="+", default=["direct", "block"])
    ap.add_argument("--filter-backends", nargs="+", default=["auto"],
                    help="'auto', 'torch' and/or 'scipy'")
    ap.add_argument("--original-commit", default="3cf55ca",
                    help="git commit for the original path, or 'none' to skip it")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trap-depth", type=float, default=4e-3, help="trap depth (T)")
    ap.add_argument("--pitch-min", type=float, default=86.5, help="degrees")
    ap.add_argument("--pitch-max", type=float, default=89.9,
                    help="degrees; draws are capped at 89.9, where the physics is numerically safe")
    ap.add_argument("--lo-mode", choices=("quarter", "centre"), default=None,
                    help="default: BatchGenFast.py's for the trap (quarter harmonic, centre bathtub)")
    ap.add_argument("--lo-freq", type=float, default=None, help="explicit LO (Hz)")
    ap.add_argument("--trap", choices=("harmonic", "bathtub"), default="harmonic")
    ap.add_argument("--coil-z", type=float, default=None,
                    help="bathtub coil half-separation (m), default 0.05")
    ap.add_argument("--background", type=float, default=1.0, help="background field (T)")
    ap.add_argument("--max-order", type=int, default=None,
                    help="maximum sideband order (default: BatchGenFast.py's, 8 harmonic, 24 bathtub)")
    ap.add_argument("--json", default=None, help="write results to this JSON file")
    import BatchGenFast
    return BatchGenFast.resolve_trap_defaults(ap.parse_args())


def load_original_package(commit, repo_root):
    """Extract CRESSignalStructure at `commit` into a temporary importable package."""
    if commit.lower() == "none":
        return None, None
    tmp = tempfile.mkdtemp(prefix="css_orig_")
    try:
        archive = subprocess.run(["git", "-C", repo_root, "archive", commit, "CRESSignalStructure"],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[original] skipped: could not extract commit {commit} ({exc})")
        shutil.rmtree(tmp, ignore_errors=True)
        return None, None
    os.rename(os.path.join(tmp, "CRESSignalStructure"), os.path.join(tmp, "CRESSignalStructure_orig"))
    sys.path.insert(0, tmp)
    pkg = importlib.import_module("CRESSignalStructure_orig")
    return pkg, tmp


def trap_kwargs(args):
    """BatchGenFast.build_trap options of the gated geometry."""
    return {'trap_type': args.trap, 'coil_z': args.coil_z, 'background': args.background}


def physics_setup(pkg, trap_depth, lo, args):
    """Field and waveguide as in BatchGenFast.py, built from the given package."""
    import BatchGenFast
    rf = importlib.import_module(pkg.__name__ + ".RealFields")
    cw = importlib.import_module(pkg.__name__ + ".CircularWaveguide")
    field_cls = rf.BathtubField if args.trap == "bathtub" else rf.HarmonicField
    trap = BatchGenFast.build_trap(trap_depth, field_cls=field_cls, **trap_kwargs(args))
    wg = cw.CircularWaveguide(radius=5e-3)
    return trap, wg, lo


def plan_lo(args):
    """LO of the gated geometry, from BatchGenFast.py's checks and convention.

    Computed once with the working tree and shared by every path, so the
    paths differ only in the generator. Refuses an untrapped pitch range or
    an aliasing carrier exactly as BatchGenFast.py does."""
    import BatchGenFast
    trap = BatchGenFast.build_trap(args.trap_depth, **trap_kwargs(args))
    wg = BatchGenFast.CircularWaveguide(radius=BatchGenFast.WAVEGUIDE_RADIUS_M)
    ranges = {'energy': (18500.0, 18600.0), 'pitch': (args.pitch_min, args.pitch_max),
              'r': (0.0, 5e-3)}
    plan = BatchGenFast.frequency_plan(trap, wg, args.trap_depth, ranges)
    lo = float(BatchGenFast.choose_lo(trap, wg, ranges, plan, 1e9, args.lo_mode, args.lo_freq))
    BatchGenFast.check_carrier_in_band(plan, lo, 1e9)
    return lo, plan


def draw_particles(pkg, n, seed, pitch_min=86.5, pitch_max=89.9):
    pa = importlib.import_module(pkg.__name__ + ".Particle")
    rng = np.random.default_rng(seed)
    ke = rng.uniform(18500, 18600, n)
    pitch = np.radians(rng.uniform(pitch_min, min(pitch_max, 89.9), n))
    r = np.sqrt(rng.uniform(0, 1, n)) * 5e-3
    th = rng.uniform(0, 2 * np.pi, n)
    phi_c, phi_a = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    parts = [pa.Particle(ke=float(ke[i]), pitchAngle=float(pitch[i]),
                         startPos=np.array([r[i] * np.cos(th[i]), r[i] * np.sin(th[i]), 0.0]))
             for i in range(n)]
    return parts, phi_c, phi_a


def run_reference(pkg, parts, phi_c, phi_a, acq, n, trap_depth, lo, args):
    """Time SignalGenerator.generate_signal from `pkg`, one event at a time."""
    sg = importlib.import_module(pkg.__name__ + ".SignalGenerator")
    sp = importlib.import_module(pkg.__name__ + ".SpectrumCalculator")
    trap, wg, lo = physics_setup(pkg, trap_depth, lo, args)
    sigs = []
    t0 = time.perf_counter()
    for i in range(n):
        gen = sg.SignalGenerator(sp.SpectrumCalculator(trap, wg, parts[i]), 1e9, lo, acq)
        sigs.append(gen.generate_signal(args.max_order, float(phi_c[i]), float(phi_a[i]))[1])
    return time.perf_counter() - t0, np.array(sigs)


def rel_l2(a, b):
    return float(np.max(np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)))


def main():
    args = parse_args()
    import torch
    torch.set_num_threads(args.threads)
    import CRESSignalStructure as css
    from CRESSignalStructure.BatchedSignalGenerator import (BatchedSynthesiser,
                                                            compute_tone_parameters)
    import multiprocessing as mp

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(css.__file__)))
    print(f"CRESSignalStructure: {css.__file__}")
    print(f"python {platform.python_version()}, torch {torch.__version__}, "
          f"numpy {np.__version__}, host {platform.node()}, cpus {os.cpu_count()}")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            sys.exit("CUDA requested but torch.cuda.is_available() is False")
        print(f"GPU: {torch.cuda.get_device_name(torch.device(args.device))}")
    try:
        print(f"load average: {os.getloadavg()}")
    except OSError:
        pass

    lo, plan = plan_lo(args)
    print(f"{args.trap} trap, background {args.background:g} T, max order {args.max_order}, "
          f"trap depth {args.trap_depth * 1e3:g} mT, pitch {args.pitch_min}-{args.pitch_max} deg, "
          f"LO {lo / 1e9:.6f} GHz, carrier {(plan['carrier_Hz'][0] - lo) / 1e6:.1f} to "
          f"{(plan['carrier_Hz'][1] - lo) / 1e6:.1f} MHz")
    orig_pkg, orig_tmp = load_original_package(args.original_commit, repo_root)
    trap, wg, lo = physics_setup(css, args.trap_depth, lo, args)
    n_pool = max(args.n_events, args.n_ref_events)
    pr = (args.pitch_min, args.pitch_max)
    parts, phi_c, phi_a = draw_particles(css, n_pool, args.seed, *pr)
    n_ref = min(args.n_ref_events, n_pool)
    pool = mp.Pool(args.n_workers) if args.n_workers > 1 else None
    results = {"args": vars(args), "runs": []}
    ok = True

    for acq in args.acq_times:
        print(f"\n=== acquisition time {acq * 1e6:.0f} us ===")
        row = {"acq_time": acq}

        if orig_pkg is not None:
            o_parts, _, _ = draw_particles(orig_pkg, n_pool, args.seed, *pr)
            dt, orig_sigs = run_reference(orig_pkg, o_parts, phi_c, phi_a, acq, n_ref,
                                          args.trap_depth, lo, args)
            row["original_ev_per_s"] = n_ref / dt
            print(f"original ({args.original_commit}) : {n_ref / dt:8.2f} ev/s  ({dt / n_ref * 1e3:.1f} ms/event)")

        dt, cached_sigs = run_reference(css, parts, phi_c, phi_a, acq, n_ref, args.trap_depth, lo, args)
        row["cached_ev_per_s"] = n_ref / dt
        print(f"cached SignalGenerator : {n_ref / dt:8.2f} ev/s  ({dt / n_ref * 1e3:.1f} ms/event)")
        if orig_pkg is not None:
            err = rel_l2(cached_sigs, orig_sigs)
            row["cached_vs_original_relL2"] = err
            row["cached_vs_original_bitwise"] = bool(np.array_equal(cached_sigs, orig_sigs))
            print(f"  cached vs original: max relL2 {err:.3e}, bitwise {row['cached_vs_original_bitwise']}")
            ok &= err <= TOL

        # Physics stage alone (shared by every batched configuration)
        t0 = time.perf_counter()
        tones = compute_tone_parameters(parts, trap, wg, 1e9, lo, acq, args.max_order, phi_c, phi_a,
                                        n_workers=args.n_workers, pool=pool)
        t_phys = (time.perf_counter() - t0) / n_pool
        row["physics_ms_per_event"] = t_phys * 1e3
        print(f"physics stage ({args.n_workers} worker(s)): {t_phys * 1e3:.1f} ms/event")

        row["batched"] = []
        for phasor in args.phasors:
            for fb in args.filter_backends:
                fb_arg = None if fb == "auto" else fb
                synth = BatchedSynthesiser(1e9, lo, acq, device=args.device, phasor=phasor,
                                           filter_backend=fb_arg)
                synth.synthesise(tones.slice(0, 1))           # warm-up
                check = synth.synthesise(tones.slice(0, n_ref))
                err = rel_l2(check, cached_sigs)
                ok &= err <= TOL
                for bs in args.batch_sizes:
                    t0 = time.perf_counter()
                    try:
                        for s in range(0, n_pool, bs):
                            synth.synthesise(tones.slice(s, s + bs))
                        if args.device.startswith("cuda"):
                            torch.cuda.synchronize()
                    except torch.cuda.OutOfMemoryError:
                        print(f"batched {phasor:6s} filter={synth.filter_backend:5s} bs={bs:4d}: out of memory")
                        row["batched"].append({"phasor": phasor, "filter": synth.filter_backend,
                                               "batch_size": bs, "oom": True})
                        torch.cuda.empty_cache()
                        continue
                    t_syn = (time.perf_counter() - t0) / n_pool
                    ev_s = 1.0 / (t_phys + t_syn)
                    rec = {"phasor": phasor, "filter": synth.filter_backend, "batch_size": bs,
                           "synthesis_ms_per_event": t_syn * 1e3, "end_to_end_ev_per_s": ev_s,
                           "max_relL2_vs_cached": err}
                    row["batched"].append(rec)
                    print(f"batched {phasor:6s} filter={synth.filter_backend:5s} bs={bs:4d}: "
                          f"synthesis {t_syn * 1e3:7.2f} ms/event, end-to-end {ev_s:8.2f} ev/s, "
                          f"max relL2 vs cached {err:.2e}")
                del synth
        results["runs"].append(row)

    if pool is not None:
        pool.close()
    if orig_tmp:
        shutil.rmtree(orig_tmp, ignore_errors=True)
    print(f"\nEquivalence (all max relL2 <= {TOL:g}): {'PASS' if ok else 'FAIL'}")
    results["equivalence_pass"] = bool(ok)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
