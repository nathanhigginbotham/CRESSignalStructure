"""
SparseSpectrum.py

Sparse storage of noiseless CRES spectra, written by the generator.

A dense 40 us record is 40,000 complex64 samples (320 kB). Its noiseless
spectrum is sparse: only bins near the carrier, the sidebands and their
leakage carry signal above a small fraction of the thermal noise. This
module keeps exactly those bins, in the on-disk layout of the detection
repository's full-band cache (CRESSignalDetection
`src/zoom_refine/fullband.py`), so a directory written here is read there
by `SparseSpectra.load` with no precompute step.

Conventions (identical to CRESSignalDetection `src/zoom_refine/physics.py`)
---------------------------------------------------------------------------
* Noise: sigma_component(T) = sqrt(k_B T (fs / 2) R) / sqrt(2) per real
  component in the time domain, R = 50 ohm, and
  fft_sigma(N, T) = sigma_component(T) sqrt(N) per real component per bin of
  the unnormalised DFT X_k = sum_n x_n exp(-2 pi i k n / N) (np.fft.fft).
* Stored spectrum: X / fft_sigma(N, T_ref), the noiseless spectrum in units
  of the per-component noise std at the reference temperature T_ref
  (default 10 K). At temperature T the signal scales by sqrt(T_ref / T).
* Kept bins: |X / fft_sigma(N, T_ref)| sqrt(T_ref / T_min) > rel_threshold,
  i.e. bins whose amplitude at the coldest temperature T_min exceeds
  rel_threshold of the per-component noise std. Every dropped bin is below
  that, and the summed dropped energy is recorded per event in units of the
  per-component noise variance at T_min (one complex noise bin carries 2).
* Labels: dataset version 'v2' (physical chirp, arange time axis), label
  definition 'centre_phys': the mean downmixed carrier plus the physical
  chirp slope times the window-centre time (N - 1) / (2 fs).

Directory layout (one output unit, e.g. one 10K-event file index)
-----------------------------------------------------------------
    index.int32   concatenated kept-bin indices (raw int32)
    values.c64    concatenated kept values (raw complex64)
    offsets.npy   (E + 1,) int64 start of each event in the two arrays
    meta.npz      fullband.META_KEYS: label_hz, fa_hz, chirp_hz_s, int_snr,
                  pitch, energy, radius, dropped_energy, max_dropped,
                  source_file, key (same names, dtypes and order)
    info.json     fullband info fields plus thresholds, FFT and noise
                  conventions and generator provenance
    truth.npz     every CRESWriter truth attribute of every event, bitwise
                  equal to the dense HDF5 attributes, plus the particle
                  draws, the stream index and the generator's chirp rate and
                  power normalisation (column -> attribute map in info.json)

Event e has the dense dataset name meta['key'][e] ('signal<k>') in the
dense file meta['source_file'][e], whether or not that dense file was
written, so sparse and dense outputs of the same seeds line up event by
event.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .CRESWriter import CRESWriter

logger = logging.getLogger(__name__)

INDEX_FILE = "index.int32"
VALUES_FILE = "values.c64"
OFFSETS_FILE = "offsets.npy"
META_FILE = "meta.npz"
INFO_FILE = "info.json"
TRUTH_FILE = "truth.npz"

# fullband.META_KEYS, in the same order
META_KEYS = ("label_hz", "fa_hz", "chirp_hz_s", "int_snr", "pitch", "energy", "radius",
             "dropped_energy", "max_dropped", "source_file", "key")
ORDERS = (-2, -1, 0, 1, 2)
DATASET_VERSION = "v2"
LABEL_DEFINITION = "centre_phys"
FORMAT_NAME = "fullband_sparse_v1"

# Constants exactly as CRESSignalDetection src/zoom_refine/physics.py, so
# the labels written here are bitwise those its precompute computes
K_B = 1.380649e-23          # J/K
E_CHARGE = 1.602176634e-19  # C
EPS0 = 8.8541878128e-12     # F/m
C_LIGHT = 299792458.0       # m/s
M_E = 9.1093837015e-31      # kg
ME_C2_EV = 510998.95000     # electron rest energy in eV

ATTR_F_MEAN_DM = "Mean downmixed cyclotron frequency [Hertz]"
ATTR_F_MEAN_ABS = "Mean cyclotron frequency [Hertz]"
ATTR_F_AXIAL = "Axial frequency [Hertz]"
ATTR_ENERGY = "Energy [eV]"
ATTR_PITCH = "Pitch angle [degrees]"
ATTR_POSITION = "Starting position [metres]"
LABEL_ATTRS = (ATTR_F_MEAN_DM, ATTR_F_MEAN_ABS, ATTR_F_AXIAL, ATTR_ENERGY)

FFT_CONVENTION = ("X_k = sum_n x_n exp(-2 pi i k n / N), unnormalised forward DFT "
                  "(np.fft.fft / torch.fft.fft) of the noiseless complex64-equivalent "
                  "record in volts, computed in float64 before the cast; stored values "
                  "are X / fft_sigma(N, T_ref) cast to complex64")
NOISE_CONVENTION = ("sigma_component(T) = sqrt(k_B T (fs / 2) R) / sqrt(2) per real "
                    "component; fft_sigma(N, T) = sigma_component(T) sqrt(N) per real "
                    "component per bin; noise at T is N(0, 1) per component after "
                    "dividing by fft_sigma(N, T), signal scales by sqrt(T_ref / T)")
THRESHOLD_RULE = ("keep bin k iff |X_k / fft_sigma(N, T_ref)| * sqrt(T_ref / temp_min) "
                  "> rel_threshold")


# ---------------------------------------------------------------------------
# Noise and label conventions
# ---------------------------------------------------------------------------

def sigma_component(system_temp: float, resistance: float = 50.0, fs: float = 1e9) -> float:
    """Per-real-component time-domain noise std (V), as in the detection repo."""
    noise_power = K_B * system_temp * (fs / 2.0)
    return float(np.sqrt(noise_power * resistance) / np.sqrt(2.0))


def fft_sigma(n_samples: int, system_temp: float, resistance: float = 50.0,
              fs: float = 1e9) -> float:
    """Per-real-component noise std of one unnormalised FFT bin."""
    return sigma_component(system_temp, resistance, fs) * np.sqrt(n_samples)


def physical_chirp_slope(energy_ev, f_c_abs_hz):
    """
    Physical carrier slope in Hz/s, d(omega)/dt / (2 pi) with gamma^3

    Written operation for operation as the detection repo's
    physics.physical_chirp_slope (via realised_chirp_slope), so the labels
    agree bitwise.
    """
    energy_ev = np.asarray(energy_ev, dtype=np.float64)
    f_c_abs_hz = np.asarray(f_c_abs_hz, dtype=np.float64)
    gamma = 1.0 + energy_ev / ME_C2_EV
    beta2 = 1.0 - 1.0 / gamma**2
    k = (E_CHARGE**2 * (2.0 * np.pi * f_c_abs_hz) ** 3 * gamma * beta2
         / (6.0 * np.pi * EPS0 * C_LIGHT) / (M_E * C_LIGHT**2))
    realised = k / np.pi
    gamma = 1.0 + energy_ev / ME_C2_EV
    return realised * gamma**2 / 2.0


def event_labels(attrs: dict, n_samples: int, fs: float) -> tuple[float, float, float]:
    """
    (label_hz, fa_hz, chirp_hz_s) of one event under 'centre_phys'

    One event at a time, on 0-d arrays, exactly as the detection repo's
    precompute evaluates them (a vectorised power could round differently).
    """
    f0 = float(attrs[ATTR_F_MEAN_DM])
    slope = float(physical_chirp_slope(attrs[ATTR_ENERGY], attrs[ATTR_F_MEAN_ABS]))
    t_ref = 0.5 * (n_samples - 1) / fs
    return f0 + slope * t_ref, float(attrs[ATTR_F_AXIAL]), slope


def truth_column(attr_key: str) -> str:
    """Column name of an HDF5 attribute key in truth.npz ('Energy [eV]' -> 'energy_ev')."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", attr_key).strip("_").lower()


# ---------------------------------------------------------------------------
# Device stage: FFT, threshold and integrated SNRs
# ---------------------------------------------------------------------------

def sparsify_batch(torch, signals, fsig: float, temp_scale: float, rel_threshold: float):
    """
    Threshold the float64 spectra of a batch of noiseless records

    Parameters
    ----------
    torch : module
    signals : Tensor (B, N) complex128
        Noiseless records in volts, on the generation device
    fsig : float
        fft_sigma(N, T_ref)
    temp_scale : float
        sqrt(T_ref / T_min)
    rel_threshold : float
        Threshold in units of the per-component noise std at T_min

    Returns
    -------
    counts : NDArray (B,) int64
        Kept bins per event
    index : NDArray int32
        Concatenated kept-bin indices, ascending within each event
    values : NDArray complex64
        Concatenated kept values X / fft_sigma(N, T_ref)
    dropped_energy, max_dropped : NDArray (B,) float64
        Summed squared and largest dropped amplitude, in noise variance and
        std units at T_min
    """
    spec = torch.fft.fft(signals, dim=-1) / fsig
    amp = spec.abs() * temp_scale
    keep = amp > rel_threshold
    dropped = torch.where(keep, torch.zeros_like(amp), amp)
    rows, cols = torch.nonzero(keep, as_tuple=True)    # row-major: ascending cols per row
    values = spec[rows, cols].to(torch.complex64)
    return (keep.sum(dim=-1).cpu().numpy().astype(np.int64),
            cols.to(torch.int32).cpu().numpy(),
            values.cpu().numpy(),
            (dropped * dropped).sum(dim=-1).cpu().numpy(),
            dropped.amax(dim=-1).cpu().numpy())


def int_snrs_batch(torch, signals, label_hz: NDArray, fa_hz: NDArray, chirp_hz_s: NDArray,
                   sigma_c: float, fs: float, chunk: int = 32) -> NDArray:
    """
    Integrated SNR of each sideband order at T_ref, as fullband computes it

    Projection of the record (in per-component noise std units) onto the
    known chirped tone at label + n f_a for n in ORDERS, with the time origin
    at the window centre: |sum_k x_k exp(-i theta_k)|^2 / (2 N). Same
    expression as the detection repo's data.channel_int_snrs.

    Returns
    -------
    NDArray (B, len(ORDERS)) float64
    """
    B, n = signals.shape
    dev = signals.device
    f64 = torch.float64
    t = (torch.arange(n, dtype=f64, device=dev) - 0.5 * (n - 1)) / fs
    out = np.empty((B, len(ORDERS)))
    lab = torch.as_tensor(np.asarray(label_hz, dtype=np.float64), device=dev)
    fa = torch.as_tensor(np.asarray(fa_hz, dtype=np.float64), device=dev)
    ch = torch.as_tensor(np.asarray(chirp_hz_s, dtype=np.float64), device=dev)
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        theta = (-math.pi * ch[s:e, None]) * (t * t)
        xd = (signals[s:e] / sigma_c) * torch.polar(torch.ones_like(theta), theta)
        for j, o in enumerate(ORDERS):
            f = lab[s:e] + o * fa[s:e]
            phi = (-2.0 * math.pi) * f[:, None] * t
            p = (xd * torch.polar(torch.ones_like(phi), phi)).sum(dim=-1)
            out[s:e, j] = (p.abs() ** 2 / (2.0 * n)).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class SparseSpectrumWriter:
    """
    Write one full-band sparse directory, event batch by event batch

    Used like CRESWriter: as a context manager, after set_global_config. The
    index and value arrays stream to disk; offsets, labels and the truth
    table are held in memory (a few MB per 10K events) and written, with
    info.json last, when the context exits without an exception. A
    directory without info.json is therefore incomplete.

    Parameters
    ----------
    out_dir : str
        Directory to write (an existing one is replaced)
    source_file : str
        Name of the corresponding dense HDF5 file (written or not); stored
        per event as meta['source_file'] and in info['files']
    device : str
        Torch device for the FFT and SNR stage (the generation device)
    system_temp_ref, temp_min, rel_threshold, resistance : float
        T_ref of the stored values (K), coldest training temperature (K),
        kept-bin threshold in noise std units at temp_min, and R (ohm)
    provenance : dict, optional
        Generator configuration (commit, seeds, ranges) for info.json
    """

    def __init__(self, out_dir: str, source_file: str, device: str = "cpu",
                 system_temp_ref: float = 10.0, temp_min: float = 2.0,
                 rel_threshold: float = 1e-2, resistance: float = 50.0,
                 provenance: Optional[dict] = None):
        if temp_min <= 0 or system_temp_ref <= 0 or rel_threshold <= 0:
            raise ValueError("Temperatures and the threshold must be positive")
        self.out_dir = Path(out_dir)
        self.source_file = str(source_file)
        self.device = device
        self.system_temp_ref = float(system_temp_ref)
        self.temp_min = float(temp_min)
        self.rel_threshold = float(rel_threshold)
        self.resistance = float(resistance)
        self.provenance = provenance or {}
        self.n_offered = 0          # events offered, i.e. the dense key counter
        self.n_skipped = 0
        self._attr_writer = CRESWriter(None)
        self._fi = self._fv = None

    def set_global_config(self, trap, waveguide, sim_config: dict):
        """Physics objects for the truth attributes, as CRESWriter.set_global_config."""
        self._attr_writer.set_global_config(trap, waveguide, sim_config)
        self.sim_config = sim_config
        self.fs = float(sim_config.get('sample_rate', 1e9))

    def __enter__(self):
        from .BatchedSignalGenerator import _import_torch
        self.torch = _import_torch()
        if self.out_dir.exists():
            # Leftover of an interrupted run: restart the unit cleanly
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True)
        self._fi = open(self.out_dir / INDEX_FILE, "wb")
        self._fv = open(self.out_dir / VALUES_FILE, "wb")
        self._offsets = [0]
        self._meta = {k: [] for k in META_KEYS}
        self._truth_rows: list[dict] = []
        self.n_samples = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fi.close()
        self._fv.close()
        if exc_type is None:
            self._finalise()

    def write_events(self, signals: NDArray, particles: Sequence, draw_indices: Sequence[int],
                     phi_c: Sequence, phi_a: Sequence, f_cyc_mean: Sequence[float],
                     f_axial: Sequence[float], chirp_rates: Sequence[float],
                     power_norms: Sequence[float]):
        """
        Append a batch of events in dense-key order

        Parameters
        ----------
        signals : NDArray (B, N) complex128
            Noiseless records in volts, exactly the float64 values the dense
            writer casts to complex64
        particles : Sequence[Particle]
        draw_indices : Sequence[int]
            Index i of each event in its (seed, file, i) stream
        phi_c, phi_a : Sequence of float or None
            Initial phases, as passed to CRESWriter.write_event
        f_cyc_mean, f_axial : Sequence[float]
            Orbit-averaged cyclotron and axial frequencies (Hz)
        chirp_rates, power_norms : Sequence[float]
            Generator chirp rate d(omega)/dt (rad/s^2) and power norm (W)
        """
        torch = self.torch
        n_ev, n = signals.shape
        if self.n_samples is None:
            self.n_samples = n
        elif n != self.n_samples:
            raise ValueError("All events in a sparse directory must have the same length")

        rows, labels = [], []
        for j, p in enumerate(particles):
            self.n_offered += 1
            attrs = self._attr_writer.event_attributes(p, phi_c[j], phi_a[j],
                                                       f_cyc_mean[j], f_axial[j])
            # The detection repo's precompute skips events whose label inputs
            # are not finite; skip the same ones (the key keeps the alignment)
            if not all(np.isfinite(float(attrs[a])) for a in LABEL_ATTRS):
                self.n_skipped += 1
                logger.warning("Sparse writer skipped signal%d (non-finite label attribute)",
                               self.n_offered)
                continue
            lab = event_labels(attrs, n, self.fs)
            row = {'source_file': Path(self.source_file).name,
                   'key': f"signal{self.n_offered}", 'draw_index': int(draw_indices[j]),
                   'gen_chirp_rate_rad_s2': float(chirp_rates[j]),
                   'gen_power_norm_w': float(power_norms[j])}
            for name, value in (getattr(p, 'draw_values', None) or {}).items():
                row[f"draw_{name}"] = value
            row['attrs'] = attrs
            rows.append((j, row, attrs))
            labels.append(lab)
        if not rows:
            return

        sel = [j for j, _, _ in rows]
        x = torch.as_tensor(np.ascontiguousarray(signals[sel]), dtype=torch.complex128,
                            device=self.device)
        fsig = fft_sigma(n, self.system_temp_ref, self.resistance, self.fs)
        scale = math.sqrt(self.system_temp_ref / self.temp_min)
        with torch.no_grad():
            counts, index, values, dropped, max_dropped = sparsify_batch(
                torch, x, fsig, scale, self.rel_threshold)
            lab = np.array(labels, dtype=np.float64)
            snrs = int_snrs_batch(torch, x, lab[:, 0], lab[:, 1], lab[:, 2],
                                  sigma_component(self.system_temp_ref, self.resistance, self.fs),
                                  self.fs)
        self._fi.write(index.tobytes())
        self._fv.write(values.tobytes())
        for k, (j, row, attrs) in enumerate(rows):
            self._offsets.append(self._offsets[-1] + int(counts[k]))
            m = self._meta
            m['label_hz'].append(labels[k][0])
            m['fa_hz'].append(labels[k][1])
            m['chirp_hz_s'].append(labels[k][2])
            m['int_snr'].append(snrs[k])
            m['pitch'].append(float(attrs[ATTR_PITCH]))
            m['energy'].append(float(attrs[ATTR_ENERGY]))
            pos = np.asarray(attrs[ATTR_POSITION], dtype=np.float64)
            m['radius'].append(float(np.hypot(pos[0], pos[1])))
            m['dropped_energy'].append(float(dropped[k]))
            m['max_dropped'].append(float(max_dropped[k]))
            m['source_file'].append(Path(self.source_file).name)
            m['key'].append(row['key'])
            self._truth_rows.append(row)

    # -- finalisation -------------------------------------------------------

    def _truth_table(self) -> tuple[dict, dict]:
        """Columns of truth.npz and the column -> HDF5 attribute key map."""
        attr_keys: list[str] = []
        for r in self._truth_rows:
            for k in r['attrs']:
                if k not in attr_keys:
                    attr_keys.append(k)
        cols, colmap = {}, {}
        n = len(self._truth_rows)
        for k in attr_keys:
            name = truth_column(k)
            vals = [r['attrs'].get(k) for r in self._truth_rows]
            if any(v is None for v in vals):
                # e.g. phases absent for some events: fill with NaN
                ref = np.asarray(next(v for v in vals if v is not None))
                vals = [np.full(ref.shape, np.nan) if v is None else v for v in vals]
            cols[name] = np.asarray(vals) if n else np.empty(0)
            colmap[name] = k
        extra = [k for k in self._truth_rows[0] if k not in ('attrs',)] if n else []
        for k in extra:
            cols[k] = np.asarray([r[k] for r in self._truth_rows])
        return cols, colmap

    def _finalise(self):
        n_ev = len(self._offsets) - 1
        n = self.n_samples or int(round(self.sim_config['acq_time'] * self.fs))
        np.save(self.out_dir / OFFSETS_FILE, np.asarray(self._offsets, dtype=np.int64))
        meta = {k: np.asarray(v) for k, v in self._meta.items()}
        if n_ev == 0:
            meta['int_snr'] = np.empty((0, len(ORDERS)))
        np.savez(self.out_dir / META_FILE, **meta)
        cols, colmap = self._truth_table()
        np.savez(self.out_dir / TRUTH_FILE, **cols)
        info = make_info(self._offsets, meta, files=[self.source_file], n_skipped=self.n_skipped,
                         system_temp_ref=self.system_temp_ref, temp_min=self.temp_min,
                         rel_threshold=self.rel_threshold, n_samples=n, fs=self.fs,
                         resistance=self.resistance)
        info.update({
            'truth_columns': colmap,
            'generator': self.provenance,
            'sim_config': {k: v for k, v in self.sim_config.items()},
        })
        with open(self.out_dir / INFO_FILE, "w") as fh:
            json.dump(info, fh, indent=2, default=_json_default)
        if n_ev:
            logger.info("Sparse: %d events, median %.0f kept bins, %.1f kB/event, "
                        "dropped energy max %.3g (noise var at %.1f K)", n_ev,
                        info['kept_bins']['median'], info['bytes_per_event_mean'] / 1e3,
                        info['dropped_energy_noise_var_units']['max'], self.temp_min)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def make_info(offsets, meta: dict, files: list, n_skipped: int, system_temp_ref: float,
              temp_min: float, rel_threshold: float, n_samples: int, fs: float,
              resistance: float) -> dict:
    """info.json contents: the fullband fields (same names) plus the conventions."""
    kept = np.diff(np.asarray(offsets, dtype=np.int64))
    dropped = np.asarray(meta['dropped_energy'], dtype=np.float64)
    n_ev = kept.size
    stat = (lambda f, a, d=0.0: float(f(a)) if a.size else d)
    return {
        "files": [str(f) for f in files], "n_events": int(n_ev), "n_skipped": int(n_skipped),
        "dataset_version": DATASET_VERSION, "label_definition": LABEL_DEFINITION,
        "orders": list(ORDERS), "system_temp_ref": system_temp_ref, "temp_min": temp_min,
        "rel_threshold": rel_threshold, "n_samples": int(n_samples), "fs": fs,
        "kept_bins": {"median": stat(np.median, kept),
                      "p95": stat(lambda a: np.percentile(a, 95), kept),
                      "max": int(kept.max()) if n_ev else 0, "mean": stat(np.mean, kept)},
        "bytes_per_event_mean": stat(np.mean, kept) * 12,
        "dropped_energy_noise_var_units": {"median": stat(np.median, dropped),
                                           "max": stat(np.max, dropped)},
        "max_dropped_amplitude_noise_std_units": stat(np.max, np.asarray(meta['max_dropped'])),
        "format": FORMAT_NAME,
        "resistance_ohm": resistance,
        "fft_convention": FFT_CONVENTION,
        "noise_convention": NOISE_CONVENTION,
        "threshold_rule": THRESHOLD_RULE,
        "fft_sigma_at_t_ref": fft_sigma(n_samples, system_temp_ref, resistance, fs),
        "index_dtype": "int32", "values_dtype": "complex64",
    }


# ---------------------------------------------------------------------------
# Reading and merging
# ---------------------------------------------------------------------------

class SparseDirectory:
    """
    Minimal reader of a sparse directory (the detection repo's
    fullband.SparseSpectra reads the same layout)
    """

    def __init__(self, path, mmap: bool = False):
        path = Path(path)
        self.path = path
        if mmap:
            self.index = np.memmap(path / INDEX_FILE, dtype=np.int32, mode="r")
            self.values = np.memmap(path / VALUES_FILE, dtype=np.complex64, mode="r")
        else:
            self.index = np.fromfile(path / INDEX_FILE, dtype=np.int32)
            self.values = np.fromfile(path / VALUES_FILE, dtype=np.complex64)
        self.offsets = np.load(path / OFFSETS_FILE)
        with np.load(path / META_FILE) as z:
            self.meta = {k: z[k] for k in z.files}
        with np.load(path / TRUTH_FILE) as z:
            self.truth = {k: z[k] for k in z.files}
        with open(path / INFO_FILE) as fh:
            self.info = json.load(fh)

    def __len__(self) -> int:
        return self.offsets.size - 1

    def event(self, e: int) -> tuple[NDArray, NDArray]:
        """(index, values) of event e."""
        s, t = self.offsets[e], self.offsets[e + 1]
        return self.index[s:t], self.values[s:t]

    def dense(self, e: int) -> NDArray:
        """Dense complex64 noiseless spectrum of event e at T_ref."""
        out = np.zeros(int(self.info["n_samples"]), dtype=np.complex64)
        idx, vals = self.event(e)
        out[idx] = vals
        return out


def merge_sparse_dirs(out_dir, dirs: Sequence, overwrite: bool = False) -> Path:
    """
    Concatenate sparse directories into one (e.g. 100 x 10K files -> 1M)

    The detection repo's trainer reads one directory per split, while the
    generator writes one per file index so runs resume and split across
    jobs. Merging is a byte concatenation of the index and value arrays,
    with shifted offsets and concatenated meta and truth tables; nothing is
    recomputed. All inputs must share thresholds, temperatures and length.

    Parameters
    ----------
    out_dir : path
        Output directory (written as out_dir.partial, then renamed)
    dirs : Sequence of paths
        Input directories, in the order their events should appear
    overwrite : bool
        Replace an existing out_dir
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists")
        shutil.rmtree(out_dir)
    partial = Path(str(out_dir) + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    infos = []
    for d in dirs:
        with open(Path(d) / INFO_FILE) as fh:
            infos.append(json.load(fh))
    if not infos:
        raise ValueError("Nothing to merge")
    keys = ("system_temp_ref", "temp_min", "rel_threshold", "n_samples", "fs",
            "dataset_version", "label_definition", "resistance_ohm")
    for d, inf in zip(dirs, infos):
        for k in keys:
            if inf.get(k) != infos[0].get(k):
                raise ValueError(f"{d}: {k} = {inf.get(k)!r} differs from {infos[0].get(k)!r}")

    offsets = [np.zeros(1, dtype=np.int64)]
    metas, truths = [], []
    total = 0
    with open(partial / INDEX_FILE, "wb") as fi, open(partial / VALUES_FILE, "wb") as fv:
        for d in dirs:
            d = Path(d)
            for name, fh in ((INDEX_FILE, fi), (VALUES_FILE, fv)):
                with open(d / name, "rb") as src:
                    shutil.copyfileobj(src, fh, length=16 << 20)
            off = np.load(d / OFFSETS_FILE)
            offsets.append(off[1:] + total)
            total += int(off[-1])
            with np.load(d / META_FILE) as z:
                metas.append({k: z[k] for k in z.files})
            with np.load(d / TRUTH_FILE) as z:
                truths.append({k: z[k] for k in z.files})
    offsets = np.concatenate(offsets)
    np.save(partial / OFFSETS_FILE, offsets)
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}
    np.savez(partial / META_FILE, **meta)
    tkeys = [k for k in truths[0] if all(k in t for t in truths)]
    np.savez(partial / TRUTH_FILE, **{k: np.concatenate([t[k] for t in truths]) for k in tkeys})
    i0 = infos[0]
    info = make_info(offsets, meta, files=[f for inf in infos for f in inf["files"]],
                     n_skipped=sum(inf["n_skipped"] for inf in infos),
                     system_temp_ref=i0["system_temp_ref"], temp_min=i0["temp_min"],
                     rel_threshold=i0["rel_threshold"], n_samples=i0["n_samples"], fs=i0["fs"],
                     resistance=i0["resistance_ohm"])
    colmap = {}
    for inf in infos:
        colmap.update(inf.get("truth_columns", {}))
    info.update({"truth_columns": {k: v for k, v in colmap.items() if k in tkeys},
                 "generator": i0.get("generator", {}), "sim_config": i0.get("sim_config", {}),
                 "merged_from": [str(Path(d)) for d in dirs]})
    with open(partial / INFO_FILE, "w") as fh:
        json.dump(info, fh, indent=2, default=_json_default)
    os.replace(partial, out_dir)
    return out_dir
