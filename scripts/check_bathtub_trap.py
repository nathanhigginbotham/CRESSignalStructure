"""
check_bathtub_trap.py

Physics checks of the BatchGenFast.py bathtub trap before bulk generation,
against the harmonic trap of the same depth. Each section writes its numbers
to one JSON report (and the field profile to a figure):

  field     on-axis |B|(z) of both traps (figure), the solved bathtub current,
            on-axis B_min and B_max, the number of local minima between the
            walls (1 means a flat minimum, not two dips), and the trapping
            threshold on axis and along the field line from r = 5 mm
  background  the background that puts the bathtub carrier where the
            harmonic trap's is (18550 eV, 88 deg, r = 2.5 mm), as in the
            detection repository's fisher_crb.py
  tones     carrier, axial frequency and downmixed band position at 86.6,
            88 and 89.5 deg, and how many sideband orders fall in the band
  sidebands fraction of signal power beyond order n (Parseval: the
            amplitudes of all orders sum to 1) at those pitches, a worst case
            over a pitch scan, and the fraction the anti-alias filter passes
            (tones weighted by |H|^4) with each trap's run LO
  cost      physics and synthesis time per event against --max-order
  jitter    relative scatter of calc_omega_axial over a fine pitch scan
            (residual from a local cubic fit), both traps
  drops     events whose physics fails over a pitch x radius x energy grid
            from 86.5 to 90 deg and r up to 5 mm

Usage
-----
    PYTHONPATH=. python scripts/check_bathtub_trap.py --outdir checks/bathtub
    PYTHONPATH=. python scripts/check_bathtub_trap.py --only field tones --background 0.996

Timings are from the machine it runs on; compare orders within one run.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import BatchGenFast as BGF  # noqa: E402
from CRESSignalStructure.BatchedSignalGenerator import (BatchedSynthesiser,  # noqa: E402
                                                        compute_tone_parameters)
from CRESSignalStructure.CircularWaveguide import CircularWaveguide  # noqa: E402
from CRESSignalStructure.Particle import Particle  # noqa: E402
from CRESSignalStructure.SignalGenerator import SignalGenerator  # noqa: E402
from CRESSignalStructure.SpectrumCalculator import SpectrumCalculator  # noqa: E402

SECTIONS = ("field", "background", "tones", "sidebands", "cost", "jitter", "drops")
REPORT_PITCHES = (86.6, 88.0, 89.5)
FS = BGF.SAMPLE_RATE_HZ


def particle(ke, pitch_deg, r):
    return Particle(ke=float(ke), startPos=np.array([float(r), 0.0, 0.0]),
                    pitchAngle=np.radians(float(pitch_deg)))


def off_axis_threshold_deg(trap, r0, n_z=4001):
    """Trapping threshold (deg) for an electron starting at radius r0 in the
    midplane: sin^2 = B(r0, 0) / max |B| along its field line up to the wall."""
    z_wall = BGF._axis_span(trap) / 2 if hasattr(trap, 'coil1') else 1.0
    z = np.linspace(0.0, z_wall, n_z)
    rho = trap.calc_rho_along_field_line(r0, z)
    b = trap.evaluate_field_magnitude(rho, 0.0, z)
    return float(np.degrees(np.arcsin(np.sqrt(b[0] / np.max(b)))))


def section_field(args, traps, out):
    """On-axis profile, current, extremes, shape and thresholds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {}
    z = np.linspace(-2 * args.coil_z, 2 * args.coil_z, 4001)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, trap in traps.items():
        b = np.asarray(trap.evaluate_field_magnitude(0.0, 0.0, z))
        b_min, b_max = BGF.on_axis_extremes(trap)
        bg = float(abs(trap.background[2]))
        # Local minima of |B| strictly between the walls (the two maxima)
        inner = (np.abs(z) < (args.coil_z if name == 'bathtub' else 1.0))
        bi = b[inner]
        n_min = int(np.sum((bi[1:-1] < bi[:-2]) & (bi[1:-1] < bi[2:])))
        res[name] = {
            'coil_current_A': BGF.trap_current(trap), 'background_T': bg,
            'on_axis_B_min_T': b_min, 'on_axis_B_max_T': b_max,
            'depth_T': b_max - b_min,
            'B_centre_minus_background_T': float(trap.evaluate_field_magnitude(0, 0, 0.0)) - bg,
            'local_minima_between_walls': n_min,
            'threshold_on_axis_deg': BGF.min_trapped_pitch_deg(trap),
            'threshold_r5mm_field_line_deg': off_axis_threshold_deg(trap, 5e-3)}
        ax.plot(z * 1e3, (b - bg) * 1e3, label=f"{name} (B_bkg {bg:.6g} T)")
    for zc in (-args.coil_z, args.coil_z):
        ax.axvline(zc * 1e3, color="0.6", ls=":", lw=1)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("|B| - B_bkg on axis [mT]")
    ax.set_title(f"On-axis field, {args.trap_depth * 1e3:g} mT depth (dotted: bathtub coils)")
    ax.legend()
    fig.tight_layout()
    path = out / "bathtub_onaxis_field.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    res['figure'] = str(path)
    return res


def matched_background(args, wg):
    """Background (T) putting the bathtub carrier on the harmonic trap's at
    18550 eV, 88 deg, r = 2.5 mm (the reference point of fisher_crb.py)."""
    harm = BGF.build_trap(args.trap_depth)
    ref = particle(18550.0, 88.0, 2.5e-3)
    target = SpectrumCalculator(harm, wg, ref).get_peak_frequency(0)
    b = 1.0
    for _ in range(4):
        trap = BGF.build_trap(args.trap_depth, trap_type='bathtub', coil_z=args.coil_z, background=b)
        b *= target / SpectrumCalculator(trap, wg, particle(18550.0, 88.0, 2.5e-3)).get_peak_frequency(0)
    trap = BGF.build_trap(args.trap_depth, trap_type='bathtub', coil_z=args.coil_z, background=b)
    f = SpectrumCalculator(trap, wg, particle(18550.0, 88.0, 2.5e-3)).get_peak_frequency(0)
    return {'matched_background_T': b, 'carrier_Hz': f, 'harmonic_carrier_Hz': target,
            'residual_Hz': f - target}


def run_plan(name, trap, wg, args):
    """Frequency plan and LO of the run box, exactly as BatchGenFast.py builds
    them (LO mode: --lo-mode, else the trap's BatchGenFast.py default)."""
    ranges = {'energy': (18500.0, 18600.0), 'pitch': (86.5, 90.0), 'r': (0.0, 5e-3)}
    plan = BGF.frequency_plan(trap, wg, args.trap_depth, ranges)
    mode = args.lo_mode or BGF.TRAP_DEFAULTS[name]['lo_mode']
    lo = float(BGF.choose_lo(trap, wg, ranges, plan, FS, mode, None))
    dm = BGF.check_carrier_in_band(plan, lo, FS)
    return plan, lo, dm


def section_tones(args, traps, wg, plans):
    res = {}
    for name, trap in traps.items():
        plan, lo, dm = plans[name]
        rows = []
        for pitch in REPORT_PITCHES:
            for r in (0.0, 2.5e-3, 5e-3):
                sc = SpectrumCalculator(trap, wg, particle(18550.0, pitch, r))
                f0 = sc.get_peak_frequency(0)
                fa = sc.get_peak_frequency(1) - f0
                n_up = int(np.floor((FS / 2 - (f0 - lo)) / fa))
                n_down = int(np.floor((FS / 2 + (f0 - lo)) / fa))
                rows.append({'pitch_deg': pitch, 'r_m': r, 'carrier_Hz': f0,
                             'downmixed_carrier_Hz': f0 - lo, 'axial_Hz': fa,
                             'z_max_m': trap.calc_z_max(particle(18550.0, pitch, r)),
                             'highest_upper_order_in_band': n_up,
                             'highest_lower_order_in_band': n_down})
        res[name] = {'lo_Hz': lo, 'lo_mode': args.lo_mode or BGF.TRAP_DEFAULTS[name]['lo_mode'],
                     'downmixed_carrier_range_Hz': list(dm),
                     'axial_range_Hz': list(plan['axial_Hz']),
                     'min_trapped_pitch_deg': plan['min_trapped_pitch_deg'], 'rows': rows}
    return res


def section_sidebands(args, traps, wg, plans):
    from scipy.signal import butter, sosfreqz
    from CRESSignalStructure.SignalGenerator import FAST_SAMPLE_FACTOR, FILTER_ORDER
    sos = butter(N=FILTER_ORDER, Wn=FS / 2, btype='low', output='sos', fs=FS * FAST_SAMPLE_FACTOR)
    orders = np.arange(-60, 61)
    cuts = (5, 8, 12, 16, 18, 20, 24, 28, 32)
    res = {}
    for name, trap in traps.items():
        lo = plans[name][1]
        rows, worst = [], {n: 0.0 for n in cuts}
        scan = list(np.round(np.linspace(86.5, 89.99, 36), 4)) + list(REPORT_PITCHES)
        for pitch in scan:
            for r in (0.0, 2.5e-3, 5e-3):
                sc = SpectrumCalculator(trap, wg, particle(18550.0, pitch, r))
                P = np.abs(sc.get_peak_amp(orders)) ** 2
                f = sc.get_peak_frequency(orders)
                beyond = {n: float(1.0 - P[np.abs(orders) <= n].sum() / P.sum()) for n in cuts}
                for n in cuts:
                    worst[n] = max(worst[n], beyond[n])
                if pitch in REPORT_PITCHES or pitch == 86.5:
                    _, h = sosfreqz(sos, worN=np.abs(f - lo), fs=FS * FAST_SAMPLE_FACTOR)
                    passed = P * np.abs(h) ** 4
                    rows.append({'pitch_deg': float(pitch), 'r_m': r, 'parseval_sum': float(P.sum()),
                                 'fraction_beyond_order': {str(n): beyond[n] for n in cuts},
                                 'fraction_passed_all_orders': float(passed.sum() / P.sum()),
                                 'fraction_passed_order_le_24':
                                     float(passed[np.abs(orders) <= 24].sum() / P.sum())})
        res[name] = {'rows': rows,
                     'worst_fraction_beyond_order_pitch_scan':
                         {str(n): v for n, v in worst.items()},
                     'scan': 'pitch 86.5 to 89.99 deg (36 points) x r 0, 2.5, 5 mm, 18550 eV'}
    return res


def section_cost(args, traps, wg, plans):
    """Physics and synthesis time per event against the maximum order."""
    import torch
    torch.set_num_threads(args.threads)
    trap = traps['bathtub']
    lo = plans['bathtub'][1]
    rng = np.random.default_rng(7)
    n = args.cost_events
    res = []
    synth = BatchedSynthesiser(FS, lo, args.acq_time, device="cpu")
    for max_order in (8, 12, 16, 20, 24, 28):
        parts = [particle(rng.uniform(18500, 18600), rng.uniform(86.5, 89.9), 5e-3 * np.sqrt(rng.uniform()))
                 for _ in range(n)]
        pc, pa = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        t0 = time.perf_counter()
        tones = compute_tone_parameters(parts, trap, wg, FS, lo, args.acq_time, max_order, pc, pa)
        t_phys = (time.perf_counter() - t0) / n
        synth.synthesise(tones.slice(0, 1))
        t0 = time.perf_counter()
        synth.synthesise(tones)
        t_syn = (time.perf_counter() - t0) / n
        res.append({'max_order': max_order, 'n_tones': 2 * (2 * max_order + 1),
                    'physics_ms_per_event': t_phys * 1e3, 'synthesis_cpu_ms_per_event': t_syn * 1e3})
        print(f"  max order {max_order:2d}: physics {t_phys * 1e3:7.1f} ms/event, "
              f"CPU synthesis {t_syn * 1e3:7.1f} ms/event", flush=True)
    return {'acq_time_s': args.acq_time, 'events': n, 'rows': res}


def _local_cubic_residual(x, y, half=7):
    """Residual of y from a cubic fitted over a sliding window of 2*half+1 points."""
    out = np.full_like(y, np.nan)
    for i in range(half, len(x) - half):
        xs, ys = x[i - half:i + half + 1] - x[i], y[i - half:i + half + 1]
        c = np.polyfit(xs, ys, 3)
        out[i] = y[i] - c[-1]
    return out


def section_jitter(args, traps, wg):
    res = {}
    grid = np.round(np.arange(86.5, 89.99 + 1e-9, args.jitter_step), 6)
    for name, trap in traps.items():
        res[name] = {}
        for r in (0.0, 2.5e-3):
            fa = np.array([trap.calc_omega_axial(particle(18550.0, p, r)) / (2 * np.pi) for p in grid])
            rel_all = np.abs(_local_cubic_residual(grid, fa)) / fa
            ok = np.isfinite(rel_all)
            rel = rel_all[ok]
            bands = {}
            for lo_p, hi_p in ((86.5, 88.0), (88.0, 89.5), (89.5, 89.9), (89.9, 90.0)):
                sel = ok & (grid >= lo_p) & (grid < hi_p)
                if sel.any():
                    bands[f"{lo_p}-{hi_p}"] = {'p95': float(np.percentile(rel_all[sel], 95)),
                                              'max': float(np.max(rel_all[sel]))}
            res[name][f"r_{r * 1e3:g}mm"] = {
                'by_pitch_band': bands,
                'n_pitches': int(len(grid)), 'step_deg': args.jitter_step,
                'median_rel_residual': float(np.median(rel)),
                'p95_rel_residual': float(np.percentile(rel, 95)),
                'max_rel_residual': float(np.max(rel)),
                'max_rel_residual_pitch_deg': float(grid[7:-7][np.nanargmax(
                    np.abs(_local_cubic_residual(grid, fa))[7:-7] / fa[7:-7])]),
                'axial_Hz_range': [float(fa.min()), float(fa.max())]}
            print(f"  {name} r={r * 1e3:g} mm: median {np.median(rel):.1e}, "
                  f"p95 {np.percentile(rel, 95):.1e}, max {np.max(rel):.1e}", flush=True)
    return res


def section_drops(args, traps, wg, plans):
    """Tone computation (the generator's worker) over a grid; count failures."""
    pitches = list(np.round(np.linspace(86.5, 89.99, 50), 5)) + [89.995, 89.998, 89.999, 89.9999, 90.0]
    radii = (0.0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3)
    res = {}
    for name, trap in traps.items():
        lo = plans[name][1]
        fails = []
        n = 0
        for ke in (18500.0, 18600.0):
            for pitch in pitches:
                for r in radii:
                    n += 1
                    try:
                        sg = SignalGenerator(SpectrumCalculator(trap, wg, particle(ke, pitch, r)),
                                             FS, lo, args.acq_time)
                        f, a, k, pn = sg.get_tone_parameters(args.max_order, 0.3, 1.1)
                        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(a))
                                and np.isfinite(k) and np.isfinite(pn)):
                            raise FloatingPointError("non-finite tone parameters")
                    except Exception as exc:
                        fails.append({'energy_eV': ke, 'pitch_deg': float(pitch), 'r_m': r,
                                      'error': repr(exc)[:160]})
        below = [f for f in fails if f['pitch_deg'] < 89.99]
        res[name] = {'n_points': n, 'n_failed': len(fails), 'n_failed_below_89p99_deg': len(below),
                     'failures': fails, 'max_order': args.max_order}
        print(f"  {name}: {len(fails)} of {n} failed ({len(below)} below 89.99 deg)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", default="checks/bathtub")
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    ap.add_argument("--trap-depth", type=float, default=BGF.DEFAULT_TRAP_DEPTH_T)
    ap.add_argument("--coil-z", type=float, default=BGF.DEFAULT_COIL_Z_M)
    ap.add_argument("--background", type=float, default=BGF.BACKGROUND_FIELD_T,
                    help="bathtub background (T); the harmonic trap keeps 1 T")
    ap.add_argument("--lo-mode", choices=("quarter", "centre"), default=None,
                    help="LO mode for both traps (default: each trap's BatchGenFast.py default)")
    ap.add_argument("--acq-time", type=float, default=40e-6)
    ap.add_argument("--max-order", type=int, default=24, help="order used by the drops section")
    ap.add_argument("--cost-events", type=int, default=16)
    ap.add_argument("--jitter-step", type=float, default=0.001, help="pitch step (deg)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    wg = CircularWaveguide(radius=BGF.WAVEGUIDE_RADIUS_M)
    traps = {'harmonic': BGF.build_trap(args.trap_depth),
             'bathtub': BGF.build_trap(args.trap_depth, trap_type='bathtub', coil_z=args.coil_z,
                                       background=args.background)}
    report = {'args': vars(args), 'css_commit': BGF._git_commit()}
    plans = None
    if set(args.only) & {"tones", "sidebands", "cost", "drops"}:
        plans = {k: run_plan(k, t, wg, args) for k, t in traps.items()}
    for sec in args.only:
        print(f"[{sec}]", flush=True)
        t0 = time.time()
        if sec == "field":
            report[sec] = section_field(args, traps, out)
        elif sec == "background":
            report[sec] = matched_background(args, wg)
        elif sec == "tones":
            report[sec] = section_tones(args, traps, wg, plans)
        elif sec == "sidebands":
            report[sec] = section_sidebands(args, traps, wg, plans)
        elif sec == "cost":
            report[sec] = section_cost(args, traps, wg, plans)
        elif sec == "jitter":
            report[sec] = section_jitter(args, traps, wg)
        elif sec == "drops":
            report[sec] = section_drops(args, traps, wg, plans)
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
    path = out / f"bathtub_checks_{'_'.join(args.only)}.json"
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
