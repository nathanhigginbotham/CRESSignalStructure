"""
Unit tests for BatchedSignalGenerator

The batched path must reproduce SignalGenerator.generate_signal event by
event to within floating-point reassociation (relative L2 <= 1e-10).
"""

import h5py
import numpy as np
import pytest
import scipy.constants as sc

torch = pytest.importorskip("torch")

from CRESSignalStructure import (CircularWaveguide, Electron, HarmonicTrap,
                                 SignalGenerator, SpectrumCalculator)
from CRESSignalStructure.RealFields import HarmonicField
from CRESSignalStructure.BatchedSignalGenerator import (
    BatchedSynthesiser, ToneBatch, compute_tone_parameters, generate_batch,
    generate_ensemble_batched)

SAMPLE_RATE = 1e9
ACQ_TIME = 4e-6
MAX_ORDER = 4
TOL = 1e-10


@pytest.fixture(scope="module")
def setup():
    r_coil = 1.25e-2
    field = HarmonicField(radius=r_coil, current=2 * 4e-3 * r_coil / sc.mu_0,
                          background=1.0)
    wg = CircularWaveguide(5e-3)
    rng = np.random.default_rng(3)
    particles = [Electron(float(rng.uniform(18500, 18600)),
                          np.array([r, 0.0, 0.0]),
                          float(np.radians(rng.uniform(86.5, 89.9))))
                 for r in (0.0, 2e-3, 4e-3)]
    phi_c = rng.uniform(0, 2 * np.pi, 3)
    phi_a = rng.uniform(0, 2 * np.pi, 3)
    f_c = SpectrumCalculator(field, wg, particles[0]).get_peak_frequency(0)
    lo = float(f_c - SAMPLE_RATE / 4)
    refs = [SignalGenerator(SpectrumCalculator(field, wg, p), SAMPLE_RATE, lo,
                            ACQ_TIME).generate_signal(MAX_ORDER, float(phi_c[i]),
                                                      float(phi_a[i]))
            for i, p in enumerate(particles)]
    return field, wg, particles, phi_c, phi_a, lo, refs


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize("phasor,filter_backend", [
    ("direct", "scipy"), ("direct", "torch"), ("block", "torch")])
def test_batched_matches_reference(setup, phasor, filter_backend):
    field, wg, particles, phi_c, phi_a, lo, refs = setup
    out = generate_batch(particles, field, wg, SAMPLE_RATE, lo, ACQ_TIME,
                         MAX_ORDER, phi_c, phi_a, device="cpu", batch_size=2,
                         phasor=phasor, filter_backend=filter_backend,
                         block_len=400)
    for (t_ref, s_ref), sig in zip(refs, out.signals):
        assert np.array_equal(out.times, t_ref)
        assert sig.shape == s_ref.shape
        assert _rel(sig, s_ref) <= TOL


def test_analytical_trap_matches_reference():
    trap, wg = HarmonicTrap(1.0, 0.5), CircularWaveguide(5e-3)
    p = Electron(18600.0, np.array([1e-4, 0.0, 0.0]), np.radians(87.0))
    lo = float(SpectrumCalculator(trap, wg, p).get_peak_frequency(0) - 200e6)
    _, ref = SignalGenerator(SpectrumCalculator(trap, wg, p), SAMPLE_RATE, lo,
                             ACQ_TIME).generate_signal(MAX_ORDER, 0.3, 1.1)
    out = generate_batch([p], trap, wg, SAMPLE_RATE, lo, ACQ_TIME, MAX_ORDER,
                         0.3, 1.1, filter_backend="torch")
    assert _rel(out.signals[0], ref) <= TOL


def test_output_order_follows_input_order(setup):
    field, wg, particles, phi_c, phi_a, lo, _ = setup
    perm = [2, 0, 1]
    a = generate_batch(particles, field, wg, SAMPLE_RATE, lo, ACQ_TIME,
                       MAX_ORDER, phi_c, phi_a, batch_size=3)
    b = generate_batch([particles[i] for i in perm], field, wg, SAMPLE_RATE, lo,
                       ACQ_TIME, MAX_ORDER, phi_c[perm], phi_a[perm], batch_size=1)
    for row, i in enumerate(perm):
        assert _rel(b.signals[row], a.signals[i]) <= 1e-14
    again = generate_batch(particles, field, wg, SAMPLE_RATE, lo, ACQ_TIME,
                           MAX_ORDER, phi_c, phi_a, batch_size=3)
    assert np.array_equal(again.signals, a.signals)


def test_ensemble_writer_index_order(setup, tmp_path):
    field, wg, particles, _, _, lo, refs = setup
    cfg = {'sample_rate': SAMPLE_RATE, 'lo_freq': lo, 'acq_time': ACQ_TIME,
           'max_order': MAX_ORDER}
    out = tmp_path / "sig.h5"
    generate_ensemble_batched(str(out), 3, lambda i: particles[i], field, wg, cfg,
                              batch_size=2, verbose=False)
    zero_phase = generate_batch(particles, field, wg, SAMPLE_RATE, lo, ACQ_TIME,
                                MAX_ORDER)
    with h5py.File(out) as f:
        assert len(f['Data']) == 3
        for i, p in enumerate(particles):
            ds = f['Data'][f'signal{i + 1}']
            assert ds.attrs['Energy [eV]'] == p.get_energy()
            assert np.array_equal(ds[()], zero_phase.signals[i].astype('complex64'))


def test_float64_enforced(setup):
    field, wg, particles, _, _, lo, _ = setup
    tones = compute_tone_parameters(particles[:1], field, wg, SAMPLE_RATE, lo,
                                    ACQ_TIME, MAX_ORDER)
    assert tones.freqs.dtype == np.float64 and tones.amps.dtype == np.complex128
    synth = BatchedSynthesiser(SAMPLE_RATE, lo, ACQ_TIME, filter_backend="torch")
    assert synth.t_dev.dtype == torch.float64
    assert synth._downmixed(tones).dtype == torch.complex128
    assert synth.synthesise(tones).dtype == np.complex128
    bad = ToneBatch(tones.freqs.astype(np.float32), tones.amps.astype(np.complex64),
                    tones.chirp_rates, tones.power_norms)
    with pytest.raises(TypeError):
        synth.synthesise(bad)
    with pytest.raises(ValueError, match="MPS"):
        BatchedSynthesiser(SAMPLE_RATE, lo, ACQ_TIME, device="mps")


def test_ensemble_writer_truth_labels_and_phases(setup, tmp_path):
    """Batched writer stores the orbit-averaged carrier, axial frequency and
    initial phases, equal to what CRESWriter computes from the trap."""
    field, wg, particles, phi_c, phi_a, lo, refs = setup
    cfg = {'sample_rate': SAMPLE_RATE, 'lo_freq': lo, 'acq_time': ACQ_TIME,
           'max_order': MAX_ORDER}
    out = tmp_path / "sig.h5"
    generate_ensemble_batched(str(out), 3, lambda i: particles[i], field, wg, cfg,
                              phase_generator=lambda i: (phi_c[i], phi_a[i]),
                              batch_size=2, verbose=False)
    with h5py.File(out) as f:
        for i, p in enumerate(particles):
            a = f['Data'][f'signal{i + 1}'].attrs
            f_mean = field.calc_omega_0(p) / (2 * np.pi)
            assert a['Mean cyclotron frequency [Hertz]'] == f_mean
            assert a['Mean downmixed cyclotron frequency [Hertz]'] == f_mean - lo
            assert a['Axial frequency [Hertz]'] == field.calc_omega_axial(p) / (2 * np.pi)
            assert a['Cyclotron phase [rad]'] == phi_c[i]
            assert a['Axial phase [rad]'] == phi_a[i]
            assert np.array_equal(f['Data'][f'signal{i + 1}'][()], refs[i][1].astype('complex64'))


def test_ensemble_writer_drops_failed_events(setup, tmp_path):
    """A particle whose physics raises is dropped; the rest are written."""
    field, wg, particles, _, _, lo, _ = setup
    cfg = {'sample_rate': SAMPLE_RATE, 'lo_freq': lo, 'acq_time': ACQ_TIME,
           'max_order': MAX_ORDER}
    untrapped = Electron(18550.0, np.array([0.0, 0.0, 0.0]), float(np.radians(10.0)))
    pool = [particles[0], untrapped, particles[1]]
    out = tmp_path / "sig.h5"
    n_failed = generate_ensemble_batched(str(out), 3, lambda i: pool[i], field, wg, cfg,
                                         batch_size=3, verbose=False)
    with h5py.File(out) as f:
        assert n_failed == 1
        assert len(f['Data']) == 2
