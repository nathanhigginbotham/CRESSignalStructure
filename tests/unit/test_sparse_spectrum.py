"""
Unit tests for the sparse spectrum output (SparseSpectrum, BatchGenFast --sparse)

The sparse directory must reconstruct the FFT of the dense record to within
the threshold, carry truth attributes bitwise equal to the dense HDF5 ones,
leave the dense output byte-for-byte unchanged, and resume per file.
"""

import filecmp
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.constants as sc

torch = pytest.importorskip("torch")

from CRESSignalStructure import CircularWaveguide, Electron
from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.BatchedSignalGenerator import generate_batch, generate_ensemble_batched
from CRESSignalStructure.SparseSpectrum import (
    INDEX_FILE, INFO_FILE, META_KEYS, OFFSETS_FILE, VALUES_FILE, SparseDirectory,
    SparseSpectrumWriter, event_labels, fft_sigma, merge_sparse_dirs, sparsify_batch)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 1e9
ACQ_TIME = 4e-6
MAX_ORDER = 4
N = int(ACQ_TIME * SAMPLE_RATE)


@pytest.fixture(scope="module")
def setup():
    r_coil = 1.25e-2
    field = HarmonicField(radius=r_coil, current=2 * 4e-3 * r_coil / sc.mu_0, background=1.0)
    wg = CircularWaveguide(5e-3)
    rng = np.random.default_rng(7)
    particles = [Electron(float(rng.uniform(18500, 18600)), np.array([r, 0.0, 0.0]),
                          float(np.radians(rng.uniform(86.5, 89.9))))
                 for r in (0.0, 1e-3, 2e-3, 4e-3)]
    phases = rng.uniform(0, 2 * np.pi, (4, 2))
    from CRESSignalStructure import SpectrumCalculator
    lo = float(SpectrumCalculator(field, wg, particles[0]).get_peak_frequency(0) - SAMPLE_RATE / 4)
    cfg = {'sample_rate': SAMPLE_RATE, 'lo_freq': lo, 'acq_time': ACQ_TIME,
           'max_order': MAX_ORDER}
    return field, wg, particles, phases, cfg


def _generate(setup, tmp_path, dense=True, sparse=True, tmin=2.0, rel=1e-2, name="u"):
    field, wg, particles, phases, cfg = setup
    tmp_path.mkdir(parents=True, exist_ok=True)
    h5 = tmp_path / f"{name}_signal.h5"
    sw = None
    if sparse:
        sw = SparseSpectrumWriter(str(tmp_path / f"{name}_sparse"), source_file=str(h5),
                                  temp_min=tmin, rel_threshold=rel, provenance={'test': True})
    generate_ensemble_batched(str(h5) if dense else None, len(particles),
                              lambda i: particles[i], field, wg, cfg,
                              phase_generator=lambda i: tuple(phases[i]),
                              batch_size=3, verbose=False, sparse_writer=sw)
    return h5, tmp_path / f"{name}_sparse"


def test_round_trip_against_float64_fft(setup, tmp_path):
    """Sparse reconstruction equals the float64 FFT to within the threshold."""
    field, wg, particles, phases, cfg = setup
    _, sdir = _generate(setup, tmp_path, dense=False)
    sp = SparseDirectory(sdir)
    ref = generate_batch(particles, field, wg, SAMPLE_RATE, cfg['lo_freq'], ACQ_TIME,
                         MAX_ORDER, phases[:, 0], phases[:, 1]).signals
    fsig = fft_sigma(N, 10.0)
    scale = math.sqrt(10.0 / 2.0)
    assert len(sp) == len(particles)
    for e in range(len(sp)):
        spec = np.fft.fft(ref[e]) / fsig
        rec = sp.dense(e).astype(np.complex128)
        idx, vals = sp.event(e)
        assert idx.dtype == np.int32 and vals.dtype == np.complex64
        assert np.all(np.diff(idx) > 0)
        diff = np.abs(spec - rec) * scale
        assert diff.max() <= 1e-2 * (1 + 1e-6)
        # kept values: complex64 rounding only
        assert np.max(np.abs(spec[idx] - rec[idx])) <= 1e-6 * np.abs(spec).max()
        dropped = np.ones(N, bool)
        dropped[idx] = False
        assert np.isclose(np.sum((np.abs(spec[dropped]) * scale) ** 2),
                          sp.meta["dropped_energy"][e], rtol=1e-6)


def test_threshold_behaviour():
    """Exactly the bins above rel / sqrt(T_ref / T_min) survive."""
    n = 64
    rng = np.random.default_rng(0)
    spec = (rng.uniform(0, 0.02, n) * np.exp(1j * rng.uniform(0, 2 * np.pi, n)))
    x = torch.as_tensor(np.fft.ifft(spec)[None, :])
    scale = math.sqrt(10.0 / 2.0)
    counts, idx, vals, dropped, max_dropped = sparsify_batch(torch, x, 1.0, scale, 1e-2)
    amp = (torch.fft.fft(x[0]).abs() * scale).numpy()
    expect = np.flatnonzero(amp > 1e-2)
    assert counts[0] == expect.size and np.array_equal(idx, expect.astype(np.int32))
    assert max_dropped[0] <= 1e-2
    assert np.isclose(dropped[0], np.sum(amp[amp <= 1e-2] ** 2))
    # Colder T_min or a lower threshold keeps a superset
    c_cold = sparsify_batch(torch, x, 1.0, math.sqrt(10.0 / 1.0), 1e-2)[0][0]
    c_low = sparsify_batch(torch, x, 1.0, scale, 5e-3)[0][0]
    assert c_cold >= counts[0] and c_low >= counts[0]


def test_threshold_options_recorded(setup, tmp_path):
    _, a = _generate(setup, tmp_path / "a", dense=False, tmin=2.0, rel=1e-2)
    _, b = _generate(setup, tmp_path / "b", dense=False, tmin=4.0, rel=2e-2)
    sa, sb = SparseDirectory(a), SparseDirectory(b)
    assert sb.info["temp_min"] == 4.0 and sb.info["rel_threshold"] == 2e-2
    assert np.all(np.diff(sb.offsets) <= np.diff(sa.offsets))
    for k in ("fft_convention", "noise_convention", "threshold_rule", "generator",
              "system_temp_ref", "dataset_version", "label_definition"):
        assert k in sa.info
    assert sa.info["generator"] == {'test': True}


def test_truth_table_matches_dense_attributes(setup, tmp_path):
    h5, sdir = _generate(setup, tmp_path)
    sp = SparseDirectory(sdir)
    colmap = sp.info["truth_columns"]
    with h5py.File(h5) as f:
        assert len(f["Data"]) == len(sp)
        for e in range(len(sp)):
            key = str(sp.meta["key"][e])
            assert key == f"signal{e + 1}"
            attrs = dict(f["Data"][key].attrs)
            assert set(colmap.values()) == set(attrs)
            for col, attr in colmap.items():
                a, b = np.asarray(sp.truth[col][e]), np.asarray(attrs[attr])
                assert a.dtype == b.dtype and np.array_equal(a, b), col
            assert tuple(float(sp.meta[k][e]) for k in ("label_hz", "fa_hz", "chirp_hz_s")) \
                == event_labels(attrs, N, SAMPLE_RATE)
    assert np.array_equal(sp.truth["draw_index"], np.arange(len(sp)))
    assert tuple(sp.meta) == META_KEYS
    assert sp.meta["int_snr"].shape == (len(sp), 5)


def test_dense_output_unchanged_by_sparse_writer(setup, tmp_path):
    """Writing the sparse output alongside leaves the dense file byte-identical."""
    h5_only, _ = _generate(setup, tmp_path / "dense", sparse=False)
    h5_both, _ = _generate(setup, tmp_path / "both")
    assert filecmp.cmp(h5_only, h5_both, shallow=False)


def test_dense_only_cli_unchanged(tmp_path):
    """BatchGenFast without --sparse writes the same bytes as with --also-dense."""
    import BatchGenFast
    common = ["--files", "1", "--events", "3", "--acq-time", "4e-6", "--max-order", "4",
              "--phase-seed", "11", "--particle-seed", "12", "--batch-size", "2"]
    BatchGenFast.main(common + ["--outdir", str(tmp_path / "d")])
    BatchGenFast.main(common + ["--outdir", str(tmp_path / "s"), "--sparse", "--also-dense"])
    assert filecmp.cmp(tmp_path / "d" / "run_000_signal.h5",
                       tmp_path / "s" / "run_000_signal.h5", shallow=False)
    assert not (tmp_path / "d" / "run_000_sparse").exists()


def _sparse_bytes(d):
    return {n: (Path(d) / n).read_bytes() for n in (INDEX_FILE, VALUES_FILE, OFFSETS_FILE)}


def test_cli_resume_and_alignment(tmp_path):
    """Finished units are skipped, stale .partial units are redone, and a
    later --also-dense run lines up event by event with the sparse units."""
    import BatchGenFast
    common = ["--files", "2", "--events", "3", "--acq-time", "4e-6", "--max-order", "4",
              "--phase-seed", "11", "--particle-seed", "12", "--batch-size", "2",
              "--outdir", str(tmp_path), "--sparse"]
    BatchGenFast.main(common)
    u0, u1 = tmp_path / "run_000_sparse", tmp_path / "run_001_sparse"
    first = {u: _sparse_bytes(u) for u in (u0, u1)}
    meta1 = SparseDirectory(u1).meta
    # Simulate an interruption of unit 1: final removed, stale partial left
    import shutil
    shutil.rmtree(u1)
    (tmp_path / "run_001_sparse.partial").mkdir()
    (tmp_path / "run_001_sparse.partial" / INDEX_FILE).write_bytes(b"junk")
    mtime0 = os.path.getmtime(u0 / INFO_FILE)
    BatchGenFast.main(common)
    assert os.path.getmtime(u0 / INFO_FILE) == mtime0        # skipped
    assert not (tmp_path / "run_001_sparse.partial").exists()
    assert _sparse_bytes(u1) == first[u1]                       # same seeds, same bytes
    for k in META_KEYS:
        assert np.array_equal(SparseDirectory(u1).meta[k], meta1[k])
    # Units differ between files (independent streams)
    assert first[u0] != first[u1]
    # Adding the dense output later: only the dense files are generated
    BatchGenFast.main(common + ["--also-dense"])
    assert os.path.getmtime(u0 / INFO_FILE) == mtime0
    sp = SparseDirectory(u1)
    fsig = fft_sigma(N, sp.info["system_temp_ref"])
    with h5py.File(tmp_path / "run_001_signal.h5") as f:
        for e in range(len(sp)):
            ds = f["Data"][str(sp.meta["key"][e])]
            assert ds.attrs["Energy [eV]"] == sp.truth["energy_ev"][e]
            spec = np.fft.fft(ds[()].astype(np.complex128)) / fsig
            diff = np.abs(spec - sp.dense(e)) * math.sqrt(10.0 / 2.0)
            assert diff.max() <= 1.01e-2
    with open(u0 / INFO_FILE) as fh:
        gen = json.load(fh)["generator"]
    assert gen["particle_seed"] == 12 and gen["file_index"] == 0 and "css_commit" in gen


def test_merge(setup, tmp_path):
    _, a = _generate(setup, tmp_path / "a")
    _, b = _generate(setup, tmp_path / "b", name="v")
    out = merge_sparse_dirs(tmp_path / "m", [a, b])
    m, sa, sb = SparseDirectory(out), SparseDirectory(a), SparseDirectory(b)
    assert len(m) == len(sa) + len(sb) == m.info["n_events"]
    for e in range(len(sa)):
        assert np.array_equal(m.dense(e), sa.dense(e))
        assert np.array_equal(m.dense(len(sa) + e), sb.dense(e))
    for k in META_KEYS:
        assert np.array_equal(m.meta[k], np.concatenate([sa.meta[k], sb.meta[k]]))
    assert list(m.truth["source_file"][[0, -1]]) == ["u_signal.h5", "v_signal.h5"]
    with pytest.raises(ValueError):
        _, c = _generate(setup, tmp_path / "c", tmin=3.0)
        merge_sparse_dirs(tmp_path / "m2", [a, c])


@pytest.mark.skipif(not os.environ.get("CRES_DETECTION_REPO"),
                    reason="set CRES_DETECTION_REPO to a CRESSignalDetection checkout")
def test_detection_repo_reader(setup, tmp_path):
    """The detection repo's fullband.SparseSpectra reads the directory as written."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.expanduser(os.environ["CRES_DETECTION_REPO"]))
    from src.zoom_refine.fullband import SparseSpectra
    _, sdir = _generate(setup, tmp_path, dense=False)
    fb, own = SparseSpectra.load(sdir), SparseDirectory(sdir)
    assert len(fb) == len(own) and fb.n_samples == N
    dense = fb.dense(np.arange(len(fb)))
    for e in range(len(own)):
        assert np.array_equal(dense[e], own.dense(e))
