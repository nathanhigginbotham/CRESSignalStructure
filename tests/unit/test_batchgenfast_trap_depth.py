"""
Unit tests for the trap-depth option of BatchGenFast.py

The default run must be unchanged (4 mT, LO at f_c(ref) - fs/4), the trap
depth must reach the coil current, LO and run config, an untrapped pitch
range or an aliasing carrier must stop the run before anything is written,
and downmixed labels must keep their sign.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.constants as sc

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import BatchGenFast  # noqa: E402
from CRESSignalStructure.CircularWaveguide import CircularWaveguide  # noqa: E402
from CRESSignalStructure.Particle import Particle  # noqa: E402
from CRESSignalStructure.RealFields import HarmonicField  # noqa: E402
from CRESSignalStructure.SpectrumCalculator import SpectrumCalculator  # noqa: E402

COMMON = ["--files", "1", "--events", "3", "--acq-time", "4e-6", "--max-order", "4",
          "--phase-seed", "11", "--particle-seed", "12", "--batch-size", "2"]


def _config(outdir):
    with open(Path(outdir) / "run_config_run_000.json") as fh:
        return json.load(fh)


@pytest.mark.parametrize("depth", [4e-3, 32e-3])
def test_threshold_matches_trap(depth):
    """The analytic on-axis threshold separates trapped from untrapped pitches."""
    trap = BatchGenFast.build_trap(depth)
    p_min = BatchGenFast.min_trapped_pitch_deg(depth)
    above = Particle(ke=18550.0, startPos=np.zeros(3), pitchAngle=np.radians(p_min + 0.01))
    below = Particle(ke=18550.0, startPos=np.zeros(3), pitchAngle=np.radians(p_min - 0.01))
    assert trap.calc_z_max(above) > 0
    with pytest.raises(ValueError):
        trap.calc_z_max(below)


def test_default_run_unchanged(tmp_path):
    """Default options give the 4 mT trap and the BatchGen.py LO, bitwise."""
    BatchGenFast.main(COMMON + ["--outdir", str(tmp_path)])
    r_coil = 1.25e-2
    trap = HarmonicField(radius=r_coil, current=2 * 4e-3 * r_coil / sc.mu_0, background=1.0)
    ref = Particle(ke=18550.0, startPos=np.zeros(3), pitchAngle=np.radians((86.5 + 90.0) / 2))
    lo = SpectrumCalculator(trap, CircularWaveguide(radius=5e-3), ref).get_peak_frequency(0) - 1e9 / 4
    cfg = _config(tmp_path)
    assert cfg["lo_freq"] == lo
    assert cfg["physics"]["trap_depth_T"] == 4e-3
    assert cfg["physics"]["coil_current_A"] == 2 * 4e-3 * r_coil / sc.mu_0
    assert cfg["physics"]["lo_mode"] == "quarter"
    with h5py.File(tmp_path / "run_000_signal.h5") as f:
        for ds in f["Data"].values():
            assert ds.attrs["LO frequency [Hertz]"] == lo


def test_untrapped_pitch_range_fails_up_front(tmp_path):
    """85 deg is untrapped at 4 mT: the run stops before writing anything."""
    with pytest.raises(SystemExit):
        BatchGenFast.main(COMMON + ["--outdir", str(tmp_path), "--pitch-min", "85"])
    assert not any(tmp_path.glob("*.h5")) and not any(tmp_path.glob("*.json"))


def test_aliasing_carrier_fails_up_front(tmp_path):
    """At 32 mT, pitch 80-90, the quarter LO would alias the carrier."""
    with pytest.raises(SystemExit):
        BatchGenFast.main(COMMON + ["--outdir", str(tmp_path), "--trap-depth", "32e-3",
                                    "--pitch-min", "80", "--pitch-max", "90"])
    assert not any(tmp_path.glob("*.h5"))


def test_broad_range_centre_lo(tmp_path):
    """32 mT with --lo-mode centre: depth recorded, carrier centred and in band."""
    BatchGenFast.main(COMMON + ["--outdir", str(tmp_path), "--trap-depth", "32e-3",
                                "--pitch-min", "80", "--pitch-max", "90", "--lo-mode", "centre"])
    phys = _config(tmp_path)["physics"]
    assert phys["trap_depth_T"] == 32e-3
    assert phys["coil_current_A"] == pytest.approx(2 * 32e-3 * 1.25e-2 / sc.mu_0, rel=1e-15)
    assert phys["min_trapped_pitch_deg"] == pytest.approx(79.6952, abs=1e-4)
    lo_c, hi_c = phys["downmixed_carrier_range_Hz"]
    assert lo_c == pytest.approx(-hi_c, rel=1e-12) and 0 < hi_c < 0.45e9
    with h5py.File(tmp_path / "run_000_signal.h5") as f:
        assert len(f["Data"]) == 3


def test_signed_downmixed_label(tmp_path):
    """An LO above the carrier gives negative downmixed labels, not their magnitude."""
    BatchGenFast.main(COMMON + ["--outdir", str(tmp_path), "--lo-freq", "27.1e9"])
    with h5py.File(tmp_path / "run_000_signal.h5") as f:
        for ds in f["Data"].values():
            a = ds.attrs
            assert a["Mean downmixed cyclotron frequency [Hertz]"] == \
                a["Mean cyclotron frequency [Hertz]"] - 27.1e9 < 0
            assert a["Downmixed cyclotron frequency [Hertz]"] < 0
