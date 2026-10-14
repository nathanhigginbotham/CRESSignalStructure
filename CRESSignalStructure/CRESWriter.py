"""
CRESWriter.py

Handles HDF5 persistence for CRES simulations.
Matches legacy format: Group 'Data' containing individual datasets 'signalX'
with metadata stored as attributes with specific unit-labeled keys.
"""

import logging
import h5py
import numpy as np
import scipy.constants as sc
from typing import Any, Union

logger = logging.getLogger(__name__)

# Internal Imports
from .Particle import Particle
from .CircularWaveguide import CircularWaveguide
from .BaseTrap import BaseTrap
from .BaseField import BaseField
from .RealFields import HarmonicField, BathtubField
from .QTNMTraps import BathtubTrap, HarmonicTrap

class CRESWriter:
    def __init__(self, filename: str, mode: str = 'w'):
        self.filename = filename
        self.mode = mode
        self.file: h5py.File | None
        self.current_index = 0

        # Global objects to query for metadata
        self._trap: BaseTrap | BaseField | None
        self._waveguide: CircularWaveguide | None
        self._config: dict[str, Any] | None

    def __enter__(self):
        self.file = h5py.File(self.filename, self.mode)
        if "Data" not in self.file:
            self.file.create_group("Data")
        logger.info("Opened HDF5 file: %s (mode='%s')", self.filename, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            logger.info("Closed HDF5 file: %s (%d events written)",
                        self.filename, self.current_index)

    def set_global_config(self, trap: Union[BaseTrap, BaseField], 
                          waveguide: CircularWaveguide, 
                          sim_config: dict[str, Any]):
        """
        Stores physics objects to calculate attributes for every event.
        """
        self._trap = trap
        self._waveguide = waveguide
        self._config = sim_config

    def _get_trap_params(self):
        """Helper to extract trap parameters."""
        params = {'i_coil [Amps]': np.nan, 'r_coil [metres]': np.nan}

        if isinstance(self._trap, HarmonicField):
            params['i_coil [Amps]'] = self._trap.coil.current
            params['r_coil [metres]'] = self._trap.coil.radius

        elif isinstance(self._trap, BathtubField):
            params['i_coil [Amps]'] = self._trap.coil1.current
            params['r_coil [metres]'] = self._trap.coil1.radius
            params['z_coil1 [metres]'] = self._trap.coil1.z
            params['z_coil2 [metres]'] = self._trap.coil2.z

        return params

    def _mean_freqs(self, particle: Particle) -> tuple:
        """
        Orbit-averaged cyclotron frequency <f_c> and axial frequency, in Hz

        <f_c> is the carrier the FFT parks on: a phase-modulated tone sits at
        <f_inst>, not at f_c(z=0). Returns (nan, nan) if the trap cannot
        evaluate the orbit (e.g. the particle is not trapped).
        """
        try:
            if isinstance(self._trap, BaseTrap):
                v = particle.get_speed()
                pa = particle.get_pitch_angle()
                return (float(self._trap.calc_omega_0(v, pa) / (2 * np.pi)),
                        float(self._trap.calc_omega_axial(v, pa) / (2 * np.pi)))
            return (float(self._trap.calc_omega_0(particle) / (2 * np.pi)),
                    float(self._trap.calc_omega_axial(particle) / (2 * np.pi)))
        except Exception:
            return float('nan'), float('nan')

    def event_attributes(self, particle: Particle,
                         phi_c: float = None,
                         phi_a: float = None,
                         f_cyc_mean: float = None,
                         f_axial: float = None) -> dict:
        """
        Truth attributes of one event, keyed and ordered exactly as
        write_event stores them.

        Needs only set_global_config (no open file), so other writers (the
        sparse spectrum writer) can store attributes that are bitwise those
        of the dense HDF5 files. Arguments as for write_event.
        """
        if getattr(self, '_config', None) is None or getattr(self, '_trap', None) is None:
            raise RuntimeError("set_global_config must be called first.")

        # --- Kinematics ---
        gamma = particle.get_gamma()
        mass = particle.get_mass()
        speed = particle.get_speed()
        pos = particle.get_position()
        pitch = particle.get_pitch_angle()
        
        v_z = speed * np.cos(pitch)
        v_perp = speed * np.sin(pitch)
        velocity_vector = np.array([v_perp, 0.0, v_z])

        # --- Fields & Frequencies ---
        # B_local at the starting position
        if isinstance(self._trap, BaseField):
            B_local = self._trap.evaluate_field_magnitude(pos[0], pos[1], pos[2])
        elif isinstance(self._trap, (HarmonicTrap, BathtubTrap)):
            B_local = self._trap.get_b0()
        else:
            B_local = 1.0 # Default fallback

        f_cyc = (sc.e * B_local) / (2 * np.pi * gamma * mass)
        
        f_lo = self._config.get('lo_freq', 0.0)
        sample_rate = self._config.get('sample_rate', 1e9)
        
        # --- Waveguide ---
        omega_cyc = 2 * np.pi * f_cyc
        if isinstance(self._waveguide, CircularWaveguide):
            impedance = self._waveguide.calc_te11_impedance(omega_cyc)
        else:
            impedance = np.nan

        # Exact legacy keys, in the order they are written to HDF5
        attrs = {}
        
        # Trap / Background
        if isinstance(self._trap, (HarmonicField, BathtubField)):
            attrs['B_bkg [Tesla]'] = abs(self._trap.background[2])
        elif isinstance(self._trap, (HarmonicTrap, BathtubTrap)):
            attrs['B_bkg [Tesla]'] = self._trap.get_b0()
        else:
            attrs['B_bkg [Tesla]'] = B_local

        attrs['Cyclotron frequency [Hertz]'] = f_cyc
        attrs['Downmixed cyclotron frequency [Hertz]'] = abs(f_cyc - f_lo)
        if f_cyc_mean is None or f_axial is None:
            f_cyc_mean, f_axial = self._mean_freqs(particle)
        attrs['Mean cyclotron frequency [Hertz]'] = f_cyc_mean
        attrs['Mean downmixed cyclotron frequency [Hertz]'] = abs(f_cyc_mean - f_lo)
        attrs['Axial frequency [Hertz]'] = f_axial
        if phi_c is not None:
            attrs['Cyclotron phase [rad]'] = phi_c
        if phi_a is not None:
            attrs['Axial phase [rad]'] = phi_a
        attrs['Energy [eV]'] = particle.get_energy()
        attrs['LO frequency [Hertz]'] = f_lo
        attrs['Pitch angle [degrees]'] = np.degrees(pitch)
        attrs['Starting position [metres]'] = pos
        attrs['Starting velocity [metres/second]'] = velocity_vector
        attrs['Time step [seconds]'] = 1.0 / sample_rate
        attrs['Waveguide impedance [Ohms]'] = impedance
        
        # Coil Params
        trap_params = self._get_trap_params()
        attrs['i_coil [Amps]'] = trap_params['i_coil [Amps]']
        attrs['r_coil [metres]'] = trap_params['r_coil [metres]']

        if 'z_coil1 [metres]' in trap_params:
            attrs['z_coil1 [metres]'] = trap_params['z_coil1 [metres]']
        if 'z_coil2 [metres]' in trap_params:
            attrs['z_coil2 [metres]'] = trap_params['z_coil2 [metres]']
        
        # WG Params
        # Handle cases where Waveguide might be a mock or different object
        if isinstance(self._waveguide, CircularWaveguide):
            attrs['r_wg [metres]'] = self._waveguide.get_radius()
        else:
            attrs['r_wg [metres]'] = 0.005 # Default 5mm
        return attrs

    def write_event(self, particle: Particle,
                    time_array: np.ndarray,
                    signal_array: np.ndarray,
                    phi_c: float = None,
                    phi_a: float = None,
                    f_cyc_mean: float = None,
                    f_axial: float = None):
        """
        Writes a single event with the exact legacy attribute structure.

        Also writes the orbit-averaged ('Mean') cyclotron frequency and the
        axial frequency, which are the observable truth labels. Callers that
        already hold them (the batched generator computes them alongside the
        tone parameters) pass f_cyc_mean and f_axial to avoid recomputing the
        orbit integrals; otherwise they are computed here. phi_c / phi_a, if
        given, are stored as the initial cyclotron and axial phases.
        """
        if self.file is None or self._config is None or self._trap is None:
            raise RuntimeError("CRESWriter must be used within a 'with' statement and set_global_config must be called first.")

        # --- SAFETY CHECK ---
        if len(time_array) != len(signal_array):
            raise ValueError(
                f"Shape Mismatch: Time array ({len(time_array)}) "
                f"does not match Signal array ({len(signal_array)})"
            )
        # ----------------------------------------------

        sig_name = f"signal{self.current_index + 1}"
        grp = self.file['Data']
        
        # 1. Write Data (Complex64)
        if np.iscomplexobj(signal_array):
            data_to_write = signal_array.astype('complex64')
        else:
            # Fallback for FFT power (real float)
            data_to_write = signal_array.astype('float32')
            
        dset = grp.create_dataset(sig_name, data=data_to_write)

        # 2. Derived physics values, set as attributes in the legacy order
        # (same HDF5 calls in the same order, so files are byte-identical
        # to those written before event_attributes was split out)
        for key, value in self.event_attributes(particle, phi_c, phi_a,
                                                f_cyc_mean, f_axial).items():
            dset.attrs[key] = value

        self.current_index += 1
        logger.debug(
            "Wrote event %s: energy=%.1f eV, pitch=%.4f rad, "
            "n_samples=%d",
            sig_name, particle.get_energy(),
            particle.get_pitch_angle(), len(signal_array)
        )

    def write_scattering_event(self, result):
        """
        Write a scattering event including scatter metadata.

        Uses the initial particle for primary attributes (matching legacy
        format) and adds scattering-specific attributes.

        Parameters
        ----------
        result : ScatteringResult
            Output from ScatteringSimulator.simulate()
        """
        if self.file is None or self._config is None or self._trap is None:
            raise RuntimeError("CRESWriter must be used within a 'with' " \
                               "statement and set_global_config must be called first.")

        initial_particle = result.particles[0]
        self.write_event(initial_particle, result.times, result.signal)

        # Add scattering metadata to the dataset just written
        sig_name = f"signal{self.current_index}"
        dset = self.file['Data'][sig_name]
        attrs = dset.attrs

        attrs['n_scatters'] = len(result.scatter_times)
        attrs['escaped'] = result.escaped

        if result.scatter_times:
            attrs['scatter_times [seconds]'] = np.array(
                result.scatter_times)
            attrs['scatter_energies [eV]'] = np.array(
                [p.get_energy() for p in result.particles])
            attrs['scatter_pitch_angles [degrees]'] = np.array(
                [np.degrees(p.get_pitch_angle())
                 for p in result.particles])

            f_lo = self._config.get('lo_freq', 0.0)
            f_cyc_arr = []
            for p in result.particles:
                gamma = p.get_gamma()
                mass = p.get_mass()
                pos = p.get_position()
                if isinstance(self._trap, BaseField):
                    B_local = self._trap.evaluate_field_magnitude(
                        pos[0], pos[1], pos[2])
                elif isinstance(self._trap, (HarmonicTrap, BathtubTrap)):
                    B_local = self._trap.get_b0()
                else:
                    B_local = 1.0
                f_cyc_arr.append(
                    (sc.e * B_local) / (2 * np.pi * gamma * mass))

            attrs['scatter_cyclotron_frequencies [Hertz]'] = np.array(
                f_cyc_arr)
            attrs['scatter_downmixed_cyclotron_frequencies [Hertz]'] = \
                np.abs(np.array(f_cyc_arr) - f_lo)

        logger.debug(
            "Wrote scattering event: n_scatters=%d, escaped=%s, "
            "n_samples=%d",
            len(result.scatter_times), result.escaped, len(result.signal)
        )