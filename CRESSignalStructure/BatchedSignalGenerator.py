"""
BatchedSignalGenerator.py

Batched, device-capable synthesis of CRES time series that reproduces
SignalGenerator.generate_signal to within floating-point reassociation.

The per-event spectral content (tone frequencies, complex amplitudes, chirp
rate and power normalisation) is computed on the CPU with exactly the same
code as SignalGenerator (SignalGenerator.get_tone_parameters). Only the
synthesis stage is batched: the sum of complex exponentials, the LO
downmix, the radiative chirp, the zero-phase Butterworth filter
(scipy.signal.sosfiltfilt with its default odd padding), the decimation and
the power and impedance scaling. All device arithmetic is in float64 /
complex128: absolute RF phases reach ~1e6-1e7 rad over an event, so float32
cannot represent them, and Apple MPS (no float64) is rejected.

Exactness notes
---------------
* Tone phases are formed exactly as the reference forms them,
  fl(fl(2 pi f) * t_n), so every tone carries the same rounding of its large
  phase argument. Folding the LO into the tone frequencies changes that
  rounding (the ulp of a 6e6-3e7 rad phase is 1e-9-4e-9 rad); measured on
  the BatchGen.py setup it moves the output by a relative L2 of 2.9e-10 at
  40 us and 1.3e-9 at 200 us, above the 1e-10 tolerance, so the LO stays a
  separate per-sample factor applied after the tone sum, as in the reference.
* The optional block-phasor scheme writes
  exp(i phi_n) = exp(i phi_b) * exp(i psi_j) * exp(i eps_n), with phi_b the
  block-start phase, psi_j = fl(w tau_j) a per-offset table and
  eps_n = (phi_n - phi_b) - psi_j. By Sterbenz's lemma both subtractions are
  exact (for the first block phi_b = 0; for later blocks phi_n < 2 phi_b),
  so eps_n carries the reference's own rounding residue (|eps| ~ 1e-9)
  and exp(i eps_n) = 1 + i eps_n - eps_n^2 / 2 to float64 precision. The
  main term is then a batched matrix product and the residue a cheap
  correction, replacing one sin/cos pair of a large argument per tone and
  sample by a few multiplies. In eager PyTorch the correction still costs
  several memory passes, so on CPU it is not faster than 'direct'; it is
  kept as an option to benchmark on GPUs.
* The device filter is the same linear time-invariant system as the cascade
  of direct-form-II-transposed biquads in scipy, evaluated chunk-wise in
  state-space form (zero-state response by a Toeplitz product, zero-input
  response from the carried state). It is exact in exact arithmetic; the
  rounding differs at the ~1e-15 level.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.fft
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, sosfilt_zi, sosfiltfilt

from .BaseField import BaseField
from .BaseTrap import BaseTrap
from .CircularWaveguide import CircularWaveguide
from .CRESWriter import CRESWriter
from .Particle import Particle
from .SignalGenerator import (FAST_SAMPLE_FACTOR, FILTER_ORDER, IMPEDANCE,
                              SignalGenerator)
from .SpectrumCalculator import SpectrumCalculator

logger = logging.getLogger(__name__)


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("BatchedSignalGenerator requires PyTorch") from exc
    return torch


# ---------------------------------------------------------------------------
# Physics stage (CPU, identical code to SignalGenerator)
# ---------------------------------------------------------------------------

@dataclass
class ToneBatch:
    """
    Per-event spectral content for a batch of events

    Attributes
    ----------
    freqs : NDArray
        Tone frequencies in Hertz, shape (n_events, n_tones)
    amps : NDArray
        Phase-shifted complex tone amplitudes, shape (n_events, n_tones)
    chirp_rates : NDArray
        Chirp rates d(omega)/dt in rad/s^2, shape (n_events,)
    power_norms : NDArray
        Power normalisation in Watts, shape (n_events,)
    """
    freqs: NDArray
    amps: NDArray
    chirp_rates: NDArray
    power_norms: NDArray

    def __len__(self) -> int:
        return self.freqs.shape[0]

    def slice(self, start: int, stop: int) -> "ToneBatch":
        return ToneBatch(self.freqs[start:stop], self.amps[start:stop],
                         self.chirp_rates[start:stop], self.power_norms[start:stop])


def _tone_worker(args) -> tuple:
    particle, trap, waveguide, sample_rate, lo_freq, acq_time, max_order, phi_c, phi_a = args
    sig_gen = SignalGenerator(SpectrumCalculator(trap, waveguide, particle),
                              sample_rate, lo_freq, acq_time)
    return sig_gen.get_tone_parameters(max_order, phi_c, phi_a)


def _broadcast_phases(phi: ArrayLike | float | None, n: int) -> NDArray:
    if phi is None:
        return np.zeros(n)
    phi = np.asarray(phi, dtype=float)
    if phi.ndim == 0:
        return np.full(n, float(phi))
    if phi.shape != (n,):
        raise ValueError(f"Phase array has shape {phi.shape}, expected ({n},)")
    return phi


def compute_tone_parameters(particles: Sequence[Particle],
                            trap: BaseTrap | BaseField,
                            waveguide: CircularWaveguide,
                            sample_rate: float, lo_freq: float, acq_time: float,
                            max_order: int,
                            phi_c: ArrayLike | float | None = None,
                            phi_a: ArrayLike | float | None = None,
                            n_workers: int = 1,
                            pool: Optional[mp.pool.Pool] = None) -> ToneBatch:
    """
    Evaluate the per-event spectral content with SignalGenerator's own code

    Parameters
    ----------
    particles : Sequence[Particle]
        Particles, one per event
    trap, waveguide
        Field (or analytical trap) and waveguide, as for SignalGenerator
    sample_rate, lo_freq, acq_time : float
        Digitiser rate (Hz), LO frequency (Hz) and acquisition time (s)
    max_order : int
        Maximum sideband order
    phi_c, phi_a : ArrayLike or float, optional
        Initial cyclotron and axial phases per event (default zero, as in
        EnsembleGenerator._worker_generate_event)
    n_workers : int
        Processes for the physics stage (1 = in-process)
    pool : multiprocessing.pool.Pool, optional
        Existing pool to use instead of creating one

    Returns
    -------
    ToneBatch
    """
    n = len(particles)
    pc, pa = _broadcast_phases(phi_c, n), _broadcast_phases(phi_a, n)
    args = [(p, trap, waveguide, sample_rate, lo_freq, acq_time, max_order,
             float(pc[i]), float(pa[i])) for i, p in enumerate(particles)]
    if pool is not None:
        results = pool.map(_tone_worker, args)
    elif n_workers > 1 and n > 1:
        with mp.Pool(processes=n_workers) as own_pool:
            results = own_pool.map(_tone_worker, args)
    else:
        results = [_tone_worker(a) for a in args]
    return ToneBatch(freqs=np.array([r[0] for r in results], dtype=np.float64),
                     amps=np.array([r[1] for r in results], dtype=np.complex128),
                     chirp_rates=np.array([r[2] for r in results], dtype=np.float64),
                     power_norms=np.array([r[3] for r in results], dtype=np.float64))


# ---------------------------------------------------------------------------
# Device IIR: chunked state-space evaluation of an SOS cascade
# ---------------------------------------------------------------------------

def _sos_state_space(sos: NDArray) -> tuple[NDArray, NDArray, NDArray, float]:
    """
    State-space (A, B, C, D) of a cascade of DF2T biquads whose state vector
    is scipy's sosfilt state, flattened section by section.
    """
    n_sec = sos.shape[0]
    n_state = 2 * n_sec

    def step(z: NDArray, x: float) -> tuple[NDArray, float]:
        z_new = np.empty(n_state)
        for s in range(n_sec):
            b0, b1, b2, _, a1, a2 = sos[s]
            y = b0 * x + z[2 * s]
            z_new[2 * s] = b1 * x - a1 * y + z[2 * s + 1]
            z_new[2 * s + 1] = b2 * x - a2 * y
            x = y
        return z_new, x

    A = np.empty((n_state, n_state))
    C = np.empty(n_state)
    for i in range(n_state):
        e = np.zeros(n_state)
        e[i] = 1.0
        A[:, i], C[i] = step(e, 0.0)
    B, D = step(np.zeros(n_state), 1.0)
    return A, B, C, D


class _ChunkedIIR:
    """
    Apply a real SOS cascade to many real signals at once, chunk by chunk

    For chunk c with input u_c and entry state x_c:
        y_c = T u_c + O x_c,   x_{c+1} = P x_c + G u_c
    where T is the lower-triangular Toeplitz matrix of the impulse response,
    O[i] = C A^i, G[:, j] = A^(L-1-j) B and P = A^L.
    """

    def __init__(self, sos: NDArray, chunk_len: int, device, torch):
        A, B, C, D = _sos_state_space(sos)
        L = int(chunk_len)
        S = A.shape[0]
        h = np.empty(L)
        h[0] = D
        O = np.empty((L, S))
        G = np.empty((S, L))
        row = C.copy()          # C A^i
        col = B.copy()          # A^i B
        for i in range(L):
            O[i] = row
            G[:, L - 1 - i] = col
            if i + 1 < L:
                h[i + 1] = row @ B
            row = row @ A
            col = A @ col
        P = np.linalg.matrix_power(A, L)
        idx = np.arange(L)
        T = np.where(idx[:, None] >= idx[None, :], h[np.abs(idx[:, None] - idx[None, :])], 0.0)

        f64 = torch.float64
        self.torch = torch
        self.L = L
        self.S = S
        self.T_t = torch.as_tensor(T.T.copy(), dtype=f64, device=device)
        self.O_t = torch.as_tensor(O.T.copy(), dtype=f64, device=device)
        self.G_t = torch.as_tensor(G.T.copy(), dtype=f64, device=device)
        self.P_t = torch.as_tensor(P.T.copy(), dtype=f64, device=device)

    def __call__(self, x, state0, rows: Optional[NDArray] = None):
        """
        Parameters
        ----------
        x : Tensor (n_signals, M) float64
        state0 : Tensor (n_signals, S) float64
        rows : NDArray, optional
            In-chunk output indices to compute (default all). Only valid
            when the caller needs the same in-chunk positions in every chunk.

        Returns
        -------
        Tensor (n_signals, Q, len(rows)) of outputs, with Q = ceil(M / L)
        """
        torch = self.torch
        n_sig, M = x.shape
        L = self.L
        Q = -(-M // L)
        if Q * L != M:
            x = torch.nn.functional.pad(x, (0, Q * L - M))
        U = x.view(n_sig, Q, L)
        T_t, O_t = self.T_t, self.O_t
        if rows is not None:
            r = torch.as_tensor(rows, device=x.device)
            T_t, O_t = T_t[:, r], O_t[:, r]
        Y = torch.matmul(U, T_t)
        g = torch.matmul(U, self.G_t)
        X = torch.empty((n_sig, Q, self.S), dtype=x.dtype, device=x.device)
        xc = state0
        for c in range(Q):
            X[:, c] = xc
            xc = torch.addmm(g[:, c], xc, self.P_t)
        Y += torch.matmul(X, O_t)
        return Y


# ---------------------------------------------------------------------------
# Synthesis stage
# ---------------------------------------------------------------------------

class BatchedSynthesiser:
    """
    Synthesise many events at once, reproducing SignalGenerator.generate_signal

    Parameters
    ----------
    sample_rate : float
        Digitiser sample rate in Hertz
    lo_freq : float
        Local oscillator frequency in Hertz
    acq_time : float
        Acquisition time in seconds
    device : str
        Torch device ('cpu', 'cuda', 'cuda:1', ...). MPS is rejected (no float64).
    phasor : str
        'direct' (one sin/cos pair per tone and sample, as the reference,
        sharing each pair between a tone and its negative-frequency partner)
        or 'block' (block-phasor scheme with exact rounding residue)
    block_len : int
        Samples per phasor block for phasor='block'
    filter_backend : str
        'torch' (chunked state-space IIR on the device) or 'scipy'
        (scipy.signal.sosfiltfilt on the CPU, bitwise the reference filter).
        Default: 'scipy' on CPU, 'torch' otherwise.
    filter_chunk : int
        Chunk length of the device IIR; must be a multiple of the
        oversampling factor so the decimated rows repeat in every chunk
    max_elements : int, optional
        Upper bound on (events x tones x samples) per synthesis chunk, which
        bounds device memory (default 2**22 on CPU, 2**25 otherwise)
    """

    def __init__(self, sample_rate: float, lo_freq: float, acq_time: float,
                 device: str = "cpu", phasor: str = "direct", block_len: int = 1000,
                 filter_backend: Optional[str] = None, filter_chunk: int = 500,
                 max_elements: Optional[int] = None):
        torch = _import_torch()
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "mps":
            raise ValueError("MPS has no float64 support; use 'cpu' or 'cuda'")
        if phasor not in ("direct", "block"):
            raise ValueError("phasor must be 'direct' or 'block'")
        if filter_backend is None:
            filter_backend = "scipy" if self.device.type == "cpu" else "torch"
        if filter_backend not in ("torch", "scipy"):
            raise ValueError("filter_backend must be 'torch' or 'scipy'")
        if filter_chunk % FAST_SAMPLE_FACTOR != 0:
            raise ValueError(f"filter_chunk must be a multiple of {FAST_SAMPLE_FACTOR}")

        self.sample_rate = float(sample_rate)
        self.lo_freq = float(lo_freq)
        self.acq_time = float(acq_time)
        self.phasor = phasor
        self.block_len = int(block_len)
        self.filter_backend = filter_backend
        if max_elements is None:
            max_elements = 2 ** 22 if self.device.type == "cpu" else 2 ** 25
        self.max_elements = int(max_elements)

        # Same expressions as SignalGenerator.generate_signal
        fast_freq = FAST_SAMPLE_FACTOR * self.sample_rate
        self.n_samples = int(self.acq_time * fast_freq)
        self.times_fast = np.arange(self.n_samples) / fast_freq
        self.times = self.times_fast[::FAST_SAMPLE_FACTOR]
        self.sos = butter(N=FILTER_ORDER, Wn=self.sample_rate / 2, btype='low',
                          output='sos', fs=self.sample_rate * FAST_SAMPLE_FACTOR)

        # Pad the time axis to whole phasor blocks; the extra samples are
        # computed and discarded
        n_blocks = -(-self.n_samples // self.block_len)
        n_pad = n_blocks * self.block_len if phasor == "block" else self.n_samples
        t_pad = np.arange(n_pad) / fast_freq
        f64 = torch.float64
        self.t_dev = torch.as_tensor(t_pad, dtype=f64, device=self.device)

        # LO and chirp phase arguments, formed as numpy forms them in the
        # reference: imag(-2j*pi*f_lo) = -(2*pi)*f_lo, and imag(0.5j*k) = 0.5*k
        self.w_lo = -(2 * math.pi) * self.lo_freq
        self.t_sq_dev = self.t_dev * self.t_dev

        # sosfiltfilt padding (padtype='odd', padlen=None)
        n_sec = self.sos.shape[0]
        ntaps = 2 * n_sec + 1
        ntaps -= min((self.sos[:, 2] == 0).sum(), (self.sos[:, 5] == 0).sum())
        self.edge = 3 * int(ntaps)
        if self.n_samples <= self.edge:
            raise ValueError("Acquisition too short for the filter padding")
        self.zi = sosfilt_zi(self.sos).reshape(-1)
        if filter_backend == "torch":
            self.iir = _ChunkedIIR(self.sos, filter_chunk, self.device, torch)
            self.zi_dev = torch.as_tensor(self.zi, dtype=f64, device=self.device)

    # -- synthesis ---------------------------------------------------------

    def _check_tones(self, tones: ToneBatch):
        if tones.freqs.dtype != np.float64 or tones.amps.dtype != np.complex128:
            raise TypeError("Tone parameters must be float64 / complex128")

    def _downmixed(self, tones: ToneBatch):
        """Complex downmixed fast-sampled signal, shape (n_events, n_samples)."""
        torch = self.torch
        dev = self.device
        n_ev, n_tone = tones.freqs.shape
        w = (2 * math.pi) * torch.as_tensor(tones.freqs, dtype=torch.float64, device=dev)
        amps = torch.as_tensor(tones.amps, dtype=torch.complex128, device=dev)
        n_tot = self.t_dev.shape[0]
        out = torch.empty((n_ev, n_tot), dtype=torch.complex128, device=dev)

        if self.phasor == "direct":
            half = n_tone // 2
            # SignalGenerator lists the positive-frequency tones and then their
            # negative partners, -f0 - n fa = -(f0 + n fa) bitwise. Then
            # fl(w t) changes sign exactly, so the negative tone's phasor is the
            # exact conjugate and only half the sin/cos pairs are needed:
            # sum_k A_k E_k + B_k conj(E_k) = R_0 + conj(R_1), with
            # R = [A; conj(B)] @ E a batched matrix product
            paired = (n_tone % 2 == 0 and
                      np.array_equal(tones.freqs[:, half:], -tones.freqs[:, :half]))
            if paired:
                w = w[:, :half]
                coeff = torch.stack((amps[:, :half], amps[:, half:].conj()), dim=1)
                n_phasor = half
            else:
                coeff = amps[:, None, :]
                n_phasor = n_tone
            step = max(1, self.max_elements // max(1, n_ev * n_phasor))
            for s0 in range(0, n_tot, step):
                t = self.t_dev[s0:s0 + step]
                theta = w[:, :, None] * t
                E = torch.polar(torch.ones_like(theta), theta)
                R = torch.matmul(coeff, E)
                out[:, s0:s0 + step] = R[:, 0] + R[:, 1].conj() if paired else R[:, 0]
        else:
            Lb = self.block_len
            n_blocks = n_tot // Lb
            tau = self.t_dev[:Lb]
            psi = w[:, :, None] * tau                       # (E, K, Lb)
            E_j = torch.polar(torch.ones_like(psi), psi)
            t_b = self.t_dev[::Lb]                           # block starts
            phi_b = w[:, :, None] * t_b                      # (E, K, nb)
            c_b = amps[:, :, None] * torch.polar(torch.ones_like(phi_b), phi_b)
            blocks_per_chunk = max(1, self.max_elements // max(1, n_ev * n_tone * Lb))
            max_eps = 0.0
            for b0 in range(0, n_blocks, blocks_per_chunk):
                b1 = min(n_blocks, b0 + blocks_per_chunk)
                nb = b1 - b0
                t = self.t_dev[b0 * Lb:b1 * Lb].view(nb, Lb)
                cb = c_b[:, :, b0:b1]                        # (E, K, nb)
                # Main term: sum_k c_kb exp(i psi_kj), a batched matrix product
                main = torch.matmul(cb.transpose(1, 2), E_j)  # (E, nb, Lb)
                # Residue: both subtractions are exact (Sterbenz), so eps is
                # the reference's own rounding of its phase argument
                eps = w[:, :, None, None] * t                # (E, K, nb, Lb)
                eps.sub_(phi_b[:, :, b0:b1, None]).sub_(psi[:, :, None, :])
                max_eps = max(max_eps, float(eps.abs().amax()))
                # exp(i eps) - 1 = i eps - eps^2 / 2 to float64 precision for
                # |eps| < 1e-6 (the cubic term is below 2e-19)
                g = torch.complex(-0.5 * eps * eps, eps)
                corr = ((cb[:, :, :, None] * g) * E_j[:, :, None, :]).sum(dim=1)
                out[:, b0 * Lb:b1 * Lb] = (main + corr).reshape(n_ev, -1)
            if max_eps > 1e-6:
                raise FloatingPointError(
                    f"Block-phasor residue {max_eps:.2e} rad too large for the "
                    "series expansion; use phasor='direct'")

        theta_lo = self.w_lo * self.t_dev
        lo = torch.polar(torch.ones_like(theta_lo), theta_lo)
        k_half = 0.5 * torch.as_tensor(tones.chirp_rates, dtype=torch.float64, device=dev)
        theta_c = k_half[:, None] * self.t_sq_dev
        chirp = torch.polar(torch.ones_like(theta_c), theta_c)
        # Same association as the reference: (sum * lo) * chirp
        out = (out * lo) * chirp
        return out[:, :self.n_samples]

    # -- filtering ---------------------------------------------------------

    def _odd_ext(self, x):
        n = self.edge
        torch = self.torch
        left = 2 * x[:, :1] - torch.flip(x[:, 1:n + 1], dims=[1])
        right = 2 * x[:, -1:] - torch.flip(x[:, -(n + 1):-1], dims=[1])
        return torch.cat((left, x, right), dim=1)

    def _filter_decimate_torch(self, x):
        """sosfiltfilt (odd padding) on the device, returning every 5th sample."""
        torch = self.torch
        n_ev, N = x.shape
        ext = self._odd_ext(x)
        M = ext.shape[1]
        # Real filter on complex data: filter real and imaginary parts
        # separately, which is what complex arithmetic with real
        # coefficients reduces to
        re = torch.view_as_real(ext).permute(0, 2, 1).reshape(2 * n_ev, M)
        zi = self.zi_dev
        L = self.iir.L
        y = self.iir(re, zi[None, :] * re[:, :1]).reshape(2 * n_ev, -1)[:, :M]
        yr = torch.flip(y, dims=[1])
        # Output m of the decimated signal is y_final[edge + 5 m], which is
        # index M - 1 - edge - 5 m of the reversed pass. With L a multiple of
        # 5 these indices fall at the same in-chunk positions in every chunk.
        r0 = (M - 1 - self.edge) % FAST_SAMPLE_FACTOR
        rows = np.arange(r0, L, FAST_SAMPLE_FACTOR)
        yb = self.iir(yr, zi[None, :] * yr[:, :1], rows=rows).reshape(2 * n_ev, -1)
        # yb holds reversed-pass indices r0, r0+5, ...; convert to forward order
        n_out = len(self.times)
        first = M - 1 - self.edge - FAST_SAMPLE_FACTOR * (n_out - 1)
        i_lo = (first - r0) // FAST_SAMPLE_FACTOR
        i_hi = (M - 1 - self.edge - r0) // FAST_SAMPLE_FACTOR
        dec = torch.flip(yb[:, i_lo:i_hi + 1], dims=[1])
        dec = dec.reshape(n_ev, 2, n_out).permute(0, 2, 1).contiguous()
        return torch.view_as_complex(dec)

    def synthesise(self, tones: ToneBatch) -> NDArray:
        """
        Synthesise a batch of events

        Parameters
        ----------
        tones : ToneBatch
            Spectral content from compute_tone_parameters

        Returns
        -------
        NDArray
            Complex128 signals in volts, shape (n_events, n_out), identical in
            layout to SignalGenerator.generate_signal's second output
        """
        self._check_tones(tones)
        torch = self.torch
        with torch.no_grad():
            x = self._downmixed(tones)
            if self.filter_backend == "scipy":
                x_np = x.cpu().numpy()
                filt = sosfiltfilt(self.sos, x_np, axis=-1) * np.sqrt(IMPEDANCE)
                dec = filt[:, ::FAST_SAMPLE_FACTOR]
            else:
                dec = (self._filter_decimate_torch(x) * math.sqrt(IMPEDANCE)).cpu().numpy()
        return dec * np.sqrt(tones.power_norms)[:, None]


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Output of generate_batch: times (n_out,), signals (n_events, n_out)."""
    times: NDArray
    signals: NDArray
    particles: list
    tones: ToneBatch
    timings: dict = field(default_factory=dict)


def generate_batch(particles: Sequence[Particle],
                   trap: BaseTrap | BaseField,
                   waveguide: CircularWaveguide,
                   sample_rate: float, lo_freq: float, acq_time: float,
                   max_order: int,
                   phi_c: ArrayLike | float | None = None,
                   phi_a: ArrayLike | float | None = None,
                   device: str = "cpu",
                   batch_size: int = 64,
                   n_workers: int = 1,
                   synthesiser: Optional[BatchedSynthesiser] = None,
                   **synth_kwargs) -> BatchResult:
    """
    Generate events end to end: particles in, signals out

    Equivalent, event by event, to
    SignalGenerator(SpectrumCalculator(trap, waveguide, p), sample_rate,
    lo_freq, acq_time).generate_signal(max_order, phi_c, phi_a).

    Parameters
    ----------
    particles : Sequence[Particle]
        Particles, one per event; output rows follow this order
    trap, waveguide, sample_rate, lo_freq, acq_time, max_order
        As for SignalGenerator
    phi_c, phi_a : ArrayLike or float, optional
        Initial phases per event (default zero)
    device : str
        Torch device for synthesis
    batch_size : int
        Events synthesised together
    n_workers : int
        Processes for the CPU physics stage
    synthesiser : BatchedSynthesiser, optional
        Reuse a synthesiser (its configuration must match)
    **synth_kwargs
        Passed to BatchedSynthesiser

    Returns
    -------
    BatchResult
    """
    particles = list(particles)
    if synthesiser is None:
        synthesiser = BatchedSynthesiser(sample_rate, lo_freq, acq_time,
                                         device=device, **synth_kwargs)
    t0 = time.perf_counter()
    tones = compute_tone_parameters(particles, trap, waveguide, sample_rate,
                                    lo_freq, acq_time, max_order, phi_c, phi_a,
                                    n_workers=n_workers)
    t1 = time.perf_counter()
    signals = np.empty((len(particles), len(synthesiser.times)), dtype=np.complex128)
    for s in range(0, len(particles), batch_size):
        signals[s:s + batch_size] = synthesiser.synthesise(tones.slice(s, s + batch_size))
    t2 = time.perf_counter()
    return BatchResult(times=synthesiser.times.copy(), signals=signals,
                       particles=particles, tones=tones,
                       timings={"physics_s": t1 - t0, "synthesis_s": t2 - t1})


def _fft_power(signal: NDArray, sample_rate: float) -> tuple[NDArray, NDArray]:
    """FFT power exactly as EnsembleGenerator._worker_generate_event computes it."""
    fft_complex = np.asarray(scipy.fft.fft(signal, norm='forward'))
    freqs = scipy.fft.fftfreq(len(signal), 1.0 / sample_rate)
    return freqs, np.abs(fft_complex)**2


def generate_ensemble_batched(output_file: str,
                              n_events: int,
                              particle_generator: Callable[[int], Particle],
                              trap: BaseTrap | BaseField,
                              waveguide: CircularWaveguide,
                              sim_config: dict,
                              fft_output_file: Optional[str] = None,
                              phase_generator: Optional[Callable[[int], tuple[float, float]]] = None,
                              device: str = "cpu",
                              batch_size: int = 64,
                              n_workers: int = 1,
                              verbose: bool = True,
                              **synth_kwargs) -> None:
    """
    Batched counterpart of EnsembleGenerator.generate_ensemble

    Particles are drawn with particle_generator(i) for i = 0..n_events-1 in
    order, exactly as generate_ensemble draws them, and events are written to
    the CRESWriter files in that index order (deterministic, unlike the
    multiprocessing path of generate_ensemble, which writes in completion
    order). Truth attributes are written by CRESWriter.write_event from the
    same particles, so they are identical.

    Parameters
    ----------
    output_file, n_events, particle_generator, trap, waveguide, sim_config,
    fft_output_file, verbose
        As for EnsembleGenerator.generate_ensemble
    phase_generator : callable, optional
        Function(index) -> (phi_c, phi_a). Default: zero phases, matching
        EnsembleGenerator._worker_generate_event.
    device : str
        Torch device for synthesis
    batch_size : int
        Events per synthesis batch (also the write granularity)
    n_workers : int
        Processes for the CPU physics stage (kept alive across batches)
    **synth_kwargs
        Passed to BatchedSynthesiser
    """
    synth = BatchedSynthesiser(sim_config['sample_rate'], sim_config['lo_freq'],
                               sim_config['acq_time'], device=device, **synth_kwargs)
    sample_rate = sim_config.get('sample_rate', 1e9)
    start = time.time()
    with ExitStack() as stack:
        writer = stack.enter_context(CRESWriter(output_file))
        writer.set_global_config(trap, waveguide, sim_config)
        fft_writer = None
        if fft_output_file:
            fft_writer = stack.enter_context(CRESWriter(fft_output_file))
            fft_writer.set_global_config(trap, waveguide, sim_config)
        pool = stack.enter_context(mp.Pool(processes=n_workers)) if n_workers > 1 else None

        done = 0
        for b0 in range(0, n_events, batch_size):
            idx = range(b0, min(n_events, b0 + batch_size))
            particles = [particle_generator(i) for i in idx]
            if phase_generator is None:
                pc = pa = None
            else:
                phases = np.array([phase_generator(i) for i in idx], dtype=float)
                pc, pa = phases[:, 0], phases[:, 1]
            tones = compute_tone_parameters(
                particles, trap, waveguide, sim_config['sample_rate'],
                sim_config['lo_freq'], sim_config['acq_time'],
                sim_config['max_order'], pc, pa, pool=pool)
            signals = synth.synthesise(tones)
            for p, sig in zip(particles, signals):
                writer.write_event(p, synth.times, sig)
                if fft_writer:
                    freqs, power = _fft_power(sig, sample_rate)
                    fft_writer.write_event(p, freqs, power)
            done += len(particles)
            if verbose:
                rate = done / (time.time() - start)
                logger.info("  Processed %d/%d events (%.1f ev/s)...", done, n_events, rate)
    logger.info("Done! Saved to %s", output_file)


def generate_uniform_ensemble_batched(output_file: str,
                                      n_events: int,
                                      trap: BaseTrap | BaseField,
                                      waveguide: CircularWaveguide,
                                      sim_config: dict,
                                      ranges: dict,
                                      fft_output_file: Optional[str] = None,
                                      verbose: bool = True,
                                      seed: Optional[int] = None,
                                      device: str = "cpu",
                                      batch_size: int = 64,
                                      n_workers: int = 1,
                                      **synth_kwargs) -> None:
    """
    Batched counterpart of EnsembleGenerator.generate_uniform_ensemble

    For the same seed the particles are identical to those of
    generate_uniform_ensemble (same generator, drawn in the same order), and
    events are written in index order.

    Parameters
    ----------
    output_file, n_events, trap, waveguide, sim_config, ranges,
    fft_output_file, verbose, seed
        As for EnsembleGenerator.generate_uniform_ensemble
    device, batch_size, n_workers, **synth_kwargs
        As for generate_ensemble_batched
    """
    from .EnsembleGenerator import make_uniform_particle_generator
    rng = np.random.default_rng(seed)
    generate_ensemble_batched(
        output_file=output_file, n_events=n_events,
        particle_generator=make_uniform_particle_generator(ranges, rng),
        trap=trap, waveguide=waveguide, sim_config=sim_config,
        fft_output_file=fft_output_file, device=device, batch_size=batch_size,
        n_workers=n_workers, verbose=verbose, **synth_kwargs)
