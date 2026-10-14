"""
merge_sparse_chunks.py

Concatenate the per-file sparse directories written by
BatchGenFast.py --sparse into one full-band directory.

The detection repository's selector trainer reads one directory per split
(`train_cache`, `val_cache`), whereas the generator writes one directory per
file index so that runs resume per file and split across jobs. Merging is a
byte concatenation of index.int32 and values.c64 with shifted offsets and
concatenated meta and truth tables; nothing is recomputed, so the merged
directory holds exactly the same values. Inputs must share thresholds,
temperatures and record length (checked).

Usage
-----
    python scripts/merge_sparse_chunks.py ~/data/sparse_1M/train \\
        ~/data/sparse_1M/train_*_sparse
    python scripts/merge_sparse_chunks.py OUT --glob "~/data/sparse_1M/train_*_sparse"

The merged directory needs as much disk as its inputs (about 45 GB for 1M
events); delete the inputs afterwards if space is short.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CRESSignalStructure.SparseSpectrum import INFO_FILE, merge_sparse_dirs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="output directory")
    ap.add_argument("dirs", nargs="*", help="input sparse directories, in order")
    ap.add_argument("--glob", default=None, help="glob for the inputs (sorted)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    dirs = list(args.dirs)
    if args.glob:
        dirs += sorted(glob.glob(os.path.expanduser(args.glob)))
    # Incomplete units (.partial, or no info.json) must never be merged
    dirs = [d for d in dirs if not d.endswith(".partial")]
    bad = [d for d in dirs if not (Path(d) / INFO_FILE).exists()]
    if bad:
        raise SystemExit(f"incomplete sparse directories: {bad}")
    if not dirs:
        raise SystemExit("no input directories")
    out = merge_sparse_dirs(args.out, dirs, overwrite=args.overwrite)
    print(f"merged {len(dirs)} directories into {out}")


if __name__ == "__main__":
    main()
