"""
Unit tests for the per-particle memoisation in BaseField and SpectrumCalculator.

The cache must be invisible: every cached quantity has to be bitwise identical
to the uncached computation.
"""

import pickle

import numpy as np
import pytest
import scipy.constants as sc

from CRESSignalStructure import CircularWaveguide, Electron, SpectrumCalculator
from CRESSignalStructure.RealFields import HarmonicField

R_COIL = 1.25e-2
I_COIL = 2 * 4e-3 * R_COIL / sc.mu_0
ORDERS = np.arange(-8, 9)


def _field():
    return HarmonicField(radius=R_COIL, current=I_COIL, background=1.0)


def _electron(pitch_deg=88.0, r=2e-3):
    return Electron(18575.0, np.array([r, 0.0, 0.0]), np.radians(pitch_deg))


def _uncached_spectrum(field, wg, particle):
    """Reference values from the private uncached implementations."""
    return (field._calc_omega_axial_uncached(particle, 50000),
            field._calc_omega_0_uncached(particle, 499),
            field._calc_t_vs_z_uncached(particle, None, 10000))


class TestBaseFieldMemoisation:
    def test_no_cache_outside_context(self):
        field = _field()
        field.calc_omega_axial(_electron())
        assert '_memo_store' not in field.__dict__

    def test_cached_values_bitwise_equal(self):
        field, wg, p = _field(), CircularWaveguide(5e-3), _electron()
        w_a, w_0, (t, z) = _uncached_spectrum(field, wg, p)
        with field.memoised() as store:
            for _ in range(2):
                assert field.calc_omega_axial(p) == w_a
                assert field.calc_omega_0(p) == w_0
                t_c, z_c = field.calc_t_vs_z(p)
                assert np.array_equal(t_c, t) and np.array_equal(z_c, z)
            assert len(store) > 0
        assert '_memo_store' not in field.__dict__

    def test_returned_arrays_are_copies(self):
        field, p = _field(), _electron()
        with field.memoised():
            t1, _ = field.calc_t_vs_z(p)
            t1[:] = 0.0
            t2, _ = field.calc_t_vs_z(p)
        assert np.any(t2 != 0.0)

    def test_mutated_field_is_not_served_stale(self):
        field, p = _field(), _electron()
        with field.memoised():
            before = field.calc_omega_axial(p)
            field.coil.current *= 1.1
            after = field.calc_omega_axial(p)
        assert after != before
        assert after == _field_with_current(I_COIL * 1.1).calc_omega_axial(p)

    def test_distinct_particles_do_not_collide(self):
        field = _field()
        p1, p2 = _electron(88.0), _electron(87.0)
        with field.memoised():
            a1 = field.calc_omega_axial(p1)
            a2 = field.calc_omega_axial(p2)
        assert a1 == field.calc_omega_axial(p1)
        assert a2 == field.calc_omega_axial(p2)
        assert a1 != a2

    def test_field_pickles_without_cache(self):
        field = _field()
        with field.memoised():
            field.calc_omega_axial(_electron())
        assert '_memo_store' not in pickle.loads(pickle.dumps(field)).__dict__


def _field_with_current(current):
    return HarmonicField(radius=R_COIL, current=current, background=1.0)


class TestSpectrumCalculatorMemoisation:
    @pytest.mark.parametrize("r", [0.0, 3e-3])
    def test_spectrum_bitwise_equal_to_fresh_instances(self, r):
        field, wg, p = _field(), CircularWaveguide(5e-3), _electron(r=r)
        cached = SpectrumCalculator(field, wg, p)
        # A fresh instance per call shares nothing, so it is the uncached path
        def fresh():
            return SpectrumCalculator(field, wg, p)
        for neg in (False, True):
            assert np.array_equal(cached.get_peak_frequency(ORDERS, neg),
                                  fresh().get_peak_frequency(ORDERS, neg))
            assert np.array_equal(cached.get_peak_amp(ORDERS, neg),
                                  fresh().get_peak_amp(ORDERS, neg))
        assert cached.get_power_norm() == fresh().get_power_norm()
        assert cached.get_peak_amp(3) == fresh().get_peak_amp(3)

    def test_negative_amps_are_conjugates(self):
        spec = SpectrumCalculator(_field(), CircularWaveguide(5e-3), _electron())
        assert np.array_equal(spec.get_peak_amp(ORDERS, True),
                              np.conjugate(spec.get_peak_amp(ORDERS, False)))
