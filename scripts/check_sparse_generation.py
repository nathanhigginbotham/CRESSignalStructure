"""
check_sparse_generation.py

Validate BatchGenFast.py --sparse against the dense output of the same seeds.

Generates N events with --sparse --also-dense (or checks an existing
directory with --outdir and --no-generate) and reports, per event:

* the largest per-bin difference between the sparse reconstruction and
  np.fft.fft of the dense complex64 record, in units of the per-component
  noise std at T_min (bounded by the threshold, expect <= --sparse-rel),
  split into kept and dropped bins;
* the summed dropped energy (noise variance units at T_min; one complex
  noise bin carries 2), recomputed and as recorded;
* bytes per event of the sparse directory against the dense file;
* that every truth-table column equals the dense HDF5 attribute bitwise,
  and that the meta labels follow from those attributes.

With --detection-repo it also imports the detection repository's
`src/zoom_refine/fullband.py` (read-only: no bytecode is written there),
loads the generator-written directory with `SparseSpectra.load` and checks
it reproduces the same dense spectra, then runs that repository's own
precompute (`build_fullband_cache`) on the dense files and compares kept
bins, values, labels and SNRs with the generator-written directory.

Usage
-----
    PYTHONPATH=. python scripts/check_sparse_generation.py --n-events 64 \\
        --detection-repo ~/GitHub/CRESSignalDetection/worktrees/zoomrefine

Exit status is non-zero if any check fails.
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from CRESSignalStructure.SparseSpectrum import (META_KEYS, SparseDirectory,  # noqa: E402
                                                event_labels, fft_sigma)


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).iterdir() if p.is_file())


def check_unit(sparse_dir: Path, dense_file: Path, failures: list) -> dict:
    """Compare one sparse directory with its dense HDF5 file."""
    sp = SparseDirectory(sparse_dir)
    info = sp.info
    t_ref, t_min, rel = info["system_temp_ref"], info["temp_min"], info["rel_threshold"]
    n = int(info["n_samples"])
    fs = float(info["fs"])
    scale = math.sqrt(t_ref / t_min)
    fsig = fft_sigma(n, t_ref, info["resistance_ohm"], fs)
    colmap = info["truth_columns"]
    max_kept, max_dropped, sum_sq, rec_dropped = [], [], [], []
    n_truth_cols = 0
    with h5py.File(dense_file, "r") as f:
        n_dense = len(f["Data"])
        for e in range(len(sp)):
            key = str(sp.meta["key"][e])
            ds = f["Data"][key]
            x = ds[()].astype(np.complex128)
            dense = np.fft.fft(x) / fsig
            idx, _ = sp.event(e)
            diff = (dense - sp.dense(e).astype(np.complex128)) * scale
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            max_kept.append(float(np.abs(diff[mask]).max()) if mask.any() else 0.0)
            max_dropped.append(float(np.abs(diff[~mask]).max()) if (~mask).any() else 0.0)
            sum_sq.append(float(np.sum(np.abs(diff[~mask]) ** 2)))
            rec_dropped.append(float(sp.meta["dropped_energy"][e]))

            # Truth table: bitwise equal to the dense attributes
            attrs = dict(ds.attrs)
            for col, attr in colmap.items():
                a, b = np.asarray(sp.truth[col][e]), np.asarray(attrs[attr])
                if a.shape != b.shape or a.dtype != b.dtype or \
                        not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                    failures.append(f"{sparse_dir.name} {key}: truth {col} {a!r} != {attr} {b!r}")
            missing = set(attrs) - set(colmap.values())
            if missing:
                failures.append(f"{sparse_dir.name} {key}: attributes missing from truth: {missing}")
            n_truth_cols = len(colmap)
            # Labels follow from the dense attributes
            lab = event_labels(attrs, n, fs)
            got = (sp.meta["label_hz"][e], sp.meta["fa_hz"][e], sp.meta["chirp_hz_s"][e])
            if tuple(map(float, got)) != lab:
                failures.append(f"{sparse_dir.name} {key}: labels {got} != {lab}")
            pos = np.asarray(attrs["Starting position [metres]"])
            if (sp.meta["energy"][e] != attrs["Energy [eV]"]
                    or sp.meta["pitch"][e] != attrs["Pitch angle [degrees]"]
                    or sp.meta["radius"][e] != np.hypot(pos[0], pos[1])):
                failures.append(f"{sparse_dir.name} {key}: energy/pitch/radius mismatch")
            if sp.truth["draw_energy_ev"][e] != attrs["Energy [eV]"]:
                failures.append(f"{sparse_dir.name} {key}: draw energy differs from attribute")
    if len(sp) + info["n_skipped"] != n_dense:
        failures.append(f"{sparse_dir.name}: {len(sp)} sparse + {info['n_skipped']} skipped "
                        f"!= {n_dense} dense events")
    worst = max(max(max_kept), max(max_dropped))
    if worst > rel * 1.001:
        failures.append(f"{sparse_dir.name}: max per-bin difference {worst:.3g} > {rel}")
    if not np.allclose(sum_sq, rec_dropped, rtol=1e-3, atol=1e-6):
        failures.append(f"{sparse_dir.name}: recorded dropped energy disagrees with recomputed")
    kept = np.diff(sp.offsets)
    return {
        "unit": sparse_dir.name, "n_events": len(sp), "n_dense": n_dense,
        "max_abs_diff_kept_bins_sigma_tmin": max(max_kept),
        "max_abs_diff_dropped_bins_sigma_tmin": max(max_dropped),
        "dropped_energy_noise_var_tmin": {"median": float(np.median(sum_sq)),
                                          "max": float(np.max(sum_sq))},
        "kept_bins": {"median": float(np.median(kept)), "p95": float(np.percentile(kept, 95)),
                      "max": int(kept.max())},
        "bytes_per_event_arrays": float(kept.mean() * 12),
        "bytes_per_event_dir": dir_bytes(sparse_dir) / len(sp),
        "bytes_per_event_dense": dense_file.stat().st_size / n_dense,
        "truth_columns_checked": n_truth_cols,
    }


def check_detection_reader(repo: Path, units: list, workdir: Path, failures: list) -> dict:
    """Load the generator output with the detection repo's own reader and precompute."""
    sys.dont_write_bytecode = True      # do not touch the detection repository
    sys.path.insert(0, str(repo))
    from src.zoom_refine.fullband import (META_KEYS as FB_META_KEYS,  # noqa: E402
                                          SparseSpectra, build_fullband_cache)
    out = {"fullband_module": str(repo / "src" / "zoom_refine" / "fullband.py"), "units": []}
    if tuple(FB_META_KEYS) != tuple(META_KEYS):
        failures.append(f"META_KEYS differ: {FB_META_KEYS} vs {META_KEYS}")
    for sparse_dir, dense_file in units:
        own = SparseDirectory(sparse_dir)
        res = {"unit": sparse_dir.name}
        for mmap in (False, True):
            fb = SparseSpectra.load(sparse_dir, mmap=mmap)
            if len(fb) != len(own):
                failures.append(f"fullband length {len(fb)} != {len(own)}")
            dense_fb = fb.dense(np.arange(len(fb)))
            for e in range(len(own)):
                if not np.array_equal(dense_fb[e], own.dense(e)):
                    failures.append(f"fullband dense() differs for event {e} (mmap={mmap})")
            li, lf = fb.labels()
            if not np.allclose(li + lf, own.meta["label_hz"] * fb.n_samples / fb.info["fs"],
                               rtol=0, atol=1e-9):
                failures.append("fullband labels() inconsistent")
        res["fullband_load_ok"] = True

        # The detection repo's precompute on the dense file, same thresholds
        info = own.info
        pre_dir = workdir / f"precomputed_{sparse_dir.name}"
        if pre_dir.exists():
            shutil.rmtree(pre_dir)
        build_fullband_cache([dense_file], pre_dir, version="v2",
                             system_temp=info["system_temp_ref"], temp_min=info["temp_min"],
                             rel_threshold=info["rel_threshold"], log=lambda *_: None)
        pre = SparseSpectra.load(pre_dir)
        if len(pre) != len(own):
            failures.append(f"precompute kept {len(pre)} events, generator {len(own)}")
        scale = math.sqrt(info["system_temp_ref"] / info["temp_min"])
        n_sym, max_val = 0, 0.0
        same_sets = 0
        for e in range(min(len(pre), len(own))):
            a = set(own.event(e)[0].tolist())
            b = set(pre.index[pre.offsets[e]:pre.offsets[e + 1]].tolist())
            n_sym += len(a ^ b)
            same_sets += a == b
            d = (own.dense(e).astype(np.complex128) - pre.dense([e])[0].astype(np.complex128))
            max_val = max(max_val, float(np.abs(d).max()) * scale)
        res["events_with_identical_kept_bins"] = f"{same_sets}/{len(own)}"
        res["kept_bin_symmetric_difference_total"] = n_sym
        res["max_value_diff_vs_precompute_sigma_tmin"] = max_val
        if max_val > 1e-4:
            failures.append(f"values differ from the precompute by {max_val:.3g} sigma")
        exact = {}
        for k in ("label_hz", "fa_hz", "chirp_hz_s", "pitch", "energy", "radius",
                  "source_file", "key"):
            exact[k] = bool(np.array_equal(own.meta[k], pre.meta[k]))
            if not exact[k]:
                failures.append(f"meta {k} differs from the precompute")
            if own.meta[k].dtype.kind != pre.meta[k].dtype.kind:
                failures.append(f"meta {k} dtype {own.meta[k].dtype} vs {pre.meta[k].dtype}")
        res["meta_bitwise_equal_to_precompute"] = exact
        rel_snr = np.abs(own.meta["int_snr"] - pre.meta["int_snr"]) / np.maximum(pre.meta["int_snr"], 1e-30)
        # SNRs of absent sidebands are ~0 and their relative error meaningless
        big = pre.meta["int_snr"] > 1e-6 * pre.meta["int_snr"].max()
        res["int_snr_max_rel_diff"] = float(rel_snr[big].max())
        if res["int_snr_max_rel_diff"] > 1e-5:
            failures.append(f"int_snr differs from the precompute by {res['int_snr_max_rel_diff']:.3g}")
        res["dropped_energy_max_rel_diff"] = float(np.max(
            np.abs(own.meta["dropped_energy"] - pre.meta["dropped_energy"])
            / np.maximum(pre.meta["dropped_energy"], 1e-30)))
        for k in ("dataset_version", "label_definition", "orders", "system_temp_ref",
                  "temp_min", "rel_threshold", "n_samples", "fs"):
            if own.info[k] != pre.info[k]:
                failures.append(f"info {k}: {own.info[k]!r} vs precompute {pre.info[k]!r}")
        out["units"].append(res)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-events", type=int, default=64, help="events per file")
    ap.add_argument("--files", type=int, default=1)
    ap.add_argument("--outdir", default=None, help="default: a temporary directory, deleted after")
    ap.add_argument("--no-generate", action="store_true", help="check an existing --outdir")
    ap.add_argument("--keep", action="store_true", help="keep the generated data")
    ap.add_argument("--prefix", default="check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--detection-repo", default=os.environ.get("CRES_DETECTION_REPO"),
                    help="checkout of CRESSignalDetection with src/zoom_refine/fullband.py")
    ap.add_argument("--json", default=None, help="write the report to this file")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="after '--': further BatchGenFast.py arguments (ranges, --sparse-rel, ...)")
    args = ap.parse_args()

    tmp = None
    if args.outdir is None:
        tmp = tempfile.mkdtemp(prefix="css_sparse_check_")
        outdir = Path(tmp)
    else:
        outdir = Path(args.outdir)
    failures: list = []
    report = {}
    try:
        if not args.no_generate:
            import BatchGenFast
            extra = [a for a in args.extra if a != "--"]
            t0 = time.time()
            BatchGenFast.main(["--files", str(args.files), "--events", str(args.n_events),
                               "--prefix", args.prefix, "--phase-seed", "1020",
                               "--particle-seed", "2020", "--device", args.device,
                               "--workers", str(args.workers), "--batch-size", str(args.batch_size),
                               "--outdir", str(outdir), "--sparse", "--also-dense", *extra])
            report["generation_s"] = time.time() - t0
        units = []
        for d in sorted(outdir.glob(f"{args.prefix}_*_sparse")):
            if d.is_dir():
                units.append((d, outdir / d.name.replace("_sparse", "_signal.h5")))
        if not units:
            raise SystemExit(f"no {args.prefix}_*_sparse directories in {outdir}")
        report["units"] = [check_unit(d, h, failures) for d, h in units]
        if args.detection_repo:
            report["detection_repo"] = check_detection_reader(
                Path(args.detection_repo).expanduser(), units, outdir, failures)
        else:
            report["detection_repo"] = "skipped (no --detection-repo)"
        report["failures"] = failures
        print(json.dumps(report, indent=1, default=str))
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(report, fh, indent=1, default=str)
    finally:
        if tmp and not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print("FAILED:\n  " + "\n  ".join(failures) if failures else "ALL CHECKS PASSED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
