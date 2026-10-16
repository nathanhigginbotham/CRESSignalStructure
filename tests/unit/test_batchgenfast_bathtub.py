"""
Unit tests for the bathtub trap option of BatchGenFast.py

The bathtub must have walls at the coils and one flat minimum between them,
its current must give the requested on-axis depth, its numerical trapping
threshold must separate trapped from untrapped electrons, the harmonic
threshold must still agree with the closed form, and a bathtub run must
record the trap in its config and file attributes while a harmonic file
stays as before.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import BatchGenFast  # noqa: E402
from CRESSignalStructure.Particle import Particle  # noqa: E402
from CRESSignalStructure.RealFields import BathtubField, HarmonicField  # noqa: E402

COMMON = ["--files", "1", "--events", "3", "--acq-time", "4e-6", "--max-order", "4",
          "--phase-seed", "11", "--particle-seed", "12", "--batch-size", "2"]


@pytest.fixture(scope="module")
def bathtub():
    return BatchGenFast.build_trap(4e-3, trap_type='bathtub', coil_z=0.05, background=1.0)


def test_bathtub_shape_and_depth(bathtub):
    """Walls at the coils, a single minimum at the centre, depth as requested."""
    assert isinstance(bathtub, BathtubField)
    b_min, b_max = BatchGenFast.on_axis_extremes(bathtub)
    assert b_max - b_min == pytest.approx(4e-3, rel=1e-9)
    assert b_min == pytest.approx(float(bathtub.evaluate_field_magnitude(0, 0, 0.0)), abs=1e-12)
    z = np.linspace(-0.05, 0.05, 2001)
    b = bathtub.evaluate_field_magnitude(0.0, 0.0, z)
    interior = (b[1:-1] < b[:-2]) & (b[1:-1] < b[2:])
    assert interior.sum() == 1 and abs(z[1:-1][interior][0]) < 1e-9
    # Both coils carry the same positive current along the +z background,
    # so the walls rise above it (two dips would mean the sign is wrong)
    assert bathtub.coil1.current == bathtub.coil2.current > 0
    assert b_min > 1.0
    # The coils at +-5 cm, R = 1.25 cm need about 81.8 A for 4 mT
    assert BatchGenFast.trap_current(bathtub) == pytest.approx(81.754, abs=1e-3)


def test_bathtub_current_independent_of_background():
    """On axis |B| = B_bkg + I g(z), so the current does not depend on B_bkg."""
    a = BatchGenFast.bathtub_current(4e-3, 0.05, 1.0)
    b = BatchGenFast.bathtub_current(4e-3, 0.05, 0.996)
    assert a == pytest.approx(b, rel=1e-9)


def test_bathtub_threshold_matches_trap(bathtub):
    """The numerical on-axis threshold separates trapped from untrapped pitches."""
    p_min = BatchGenFast.min_trapped_pitch_deg(bathtub)
    assert 86.3 < p_min < 86.45
    above = Particle(ke=18550.0, startPos=np.zeros(3), pitchAngle=np.radians(p_min + 0.01))
    below = Particle(ke=18550.0, startPos=np.zeros(3), pitchAngle=np.radians(p_min - 0.01))
    assert 0 < bathtub.calc_z_max(above) < 0.05
    with pytest.raises(ValueError):
        bathtub.calc_z_max(below)


@pytest.mark.parametrize("depth", [4e-3, 32e-3])
def test_harmonic_numeric_threshold_matches_closed_form(depth):
    trap = BatchGenFast.build_trap(depth)
    assert BatchGenFast.min_trapped_pitch_deg(trap) == pytest.approx(
        BatchGenFast.min_trapped_pitch_deg(depth), abs=1e-5)


def test_build_trap_legacy_calls():
    """The positional depth and field_cls keyword still build the harmonic trap."""
    trap = BatchGenFast.build_trap(4e-3, field_cls=HarmonicField)
    assert isinstance(trap, HarmonicField)
    assert trap.coil.current == BatchGenFast.coil_current(4e-3)
    with pytest.raises(ValueError):
        BatchGenFast.build_trap(4e-3, trap_type='octupole')


def test_bathtub_run_records_trap(tmp_path):
    BatchGenFast.main(COMMON + ["--outdir", str(tmp_path), "--trap", "bathtub",
                                "--background", "0.996", "--coil-z", "0.05"])
    with open(tmp_path / "run_config_run_000.json") as fh:
        cfg = json.load(fh)
    phys = cfg["physics"]
    assert cfg["trap"] == "bathtub" and phys["trap_type"] == "bathtub"
    assert phys["coil_z_m"] == [-0.05, 0.05] and phys["background_field_T"] == 0.996
    assert phys["on_axis_B_max_T"] - phys["on_axis_B_min_T"] == pytest.approx(4e-3, rel=1e-9)
    with h5py.File(tmp_path / "run_000_signal.h5") as f:
        assert f.attrs["trap_type"] == "bathtub"
        assert f.attrs["coil_z [metres]"] == 0.05
        assert f.attrs["B_bkg [Tesla]"] == 0.996
        assert len(f["Data"]) == 3
        for ds in f["Data"].values():
            assert ds.attrs["z_coil1 [metres]"] == -0.05
            assert ds.attrs["z_coil2 [metres]"] == 0.05
            assert ds.attrs["B_bkg [Tesla]"] == 0.996
            assert ds.attrs["i_coil [Amps]"] == phys["coil_current_A"]


def test_harmonic_file_has_no_trap_attrs(tmp_path):
    """Harmonic files are left as before: no root attributes are added."""
    BatchGenFast.main(COMMON + ["--outdir", str(tmp_path)])
    with h5py.File(tmp_path / "run_000_signal.h5") as f:
        assert "trap_type" not in f.attrs and len(f.attrs) == 0


def test_bathtub_untrapped_range_fails_up_front(tmp_path):
    """86.3 deg is below the bathtub's on-axis threshold: nothing is written."""
    with pytest.raises(SystemExit):
        BatchGenFast.main(COMMON + ["--outdir", str(tmp_path), "--trap", "bathtub",
                                    "--pitch-min", "86.3"])
    assert not any(tmp_path.glob("*.h5")) and not any(tmp_path.glob("*.json"))


@pytest.mark.parametrize("trap,order,mode", [("harmonic", 8, "quarter"), ("bathtub", 24, "centre")])
def test_trap_defaults(trap, order, mode):
    """Unset --max-order and --lo-mode follow the trap; explicit values win."""
    import argparse
    args = BatchGenFast.resolve_trap_defaults(
        argparse.Namespace(trap=trap, max_order=None, lo_mode=None))
    assert (args.max_order, args.lo_mode) == (order, mode)
    args = BatchGenFast.resolve_trap_defaults(
        argparse.Namespace(trap=trap, max_order=5, lo_mode="quarter"))
    assert (args.max_order, args.lo_mode) == (5, "quarter")
