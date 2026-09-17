import time

import numpy as np
import pywt
from dataclasses import dataclass
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import savgol_filter, firwin, filtfilt
from scipy.ndimage import gaussian_filter1d


# =====================================================================
# Configs (verbatim from config.py)
# =====================================================================

@dataclass
class ClusteringConfig:
    n_components: int = 30
    n_comp_clu: int = 10
    n_clusters: int = 5
    n_subclusters: int = 10
    thres_dist: float = 0.4
    standardize: bool = True
    method_linkage: str = "ward"
    method_fclust: str = "maxclust"


@dataclass
class cwtReducerConfig:
    cluster_label: str = "sublabels"
    cluster_freq: str = "subfreqs"
    # reduce_method: srt = "average"
    cutoff_freq: tuple = (0, 400)
    slow_upthres: float = 2.0
    fast_upthres: float = 2.5
    # keep bands above 75 hz complex as the original did; false takes the real part first
    complex_bands: bool = False


@dataclass
class thresConfig:
    thres_type: str = "hard"
    # soft attenuation outside event windows by band top frequency: <5, 5-30, 30-80, >=80 hz
    soft_levels: tuple = (0.7, 0.5, 0.2, 0.1)
    # hard attenuation outside event windows for every band
    hard_floor: float = 0.05
    # mask smoothing sigma in samples for bands up to 120 hz / above
    soft_sigma: tuple = (10, 3)
    hard_sigma: tuple = (10, 1)


# the spatial jedi archive's soft levels (lab attenuation above 80 hz)
UPSTREAM_SOFT_LEVELS = (0.7, 0.5, 0.2, 0.01)


# =====================================================================
# Helper: trigger_detect (verbatim from signal_utils.py)
# =====================================================================

def trigger_detect(signal, up_thresh, down_thresh):
    signal = np.array(signal).flatten()
    len_signal = len(signal)
    pre_val = np.concatenate(([(up_thresh + down_thresh) / 2], signal[:-1]))

    up_crossings = np.where((pre_val < up_thresh) & (signal >= up_thresh))[0]
    down_crossings = np.where((pre_val > down_thresh) & (signal <= down_thresh))[0]

    up_down_unsort = np.concatenate((up_crossings, down_crossings))
    type_unsort = np.concatenate((np.ones(len(up_crossings)), -np.ones(len(down_crossings))))

    sorted_indices = np.argsort(up_down_unsort)
    up_down_sort = up_down_unsort[sorted_indices]
    type_sort = type_unsort[sorted_indices]

    prev_type = np.concatenate(([np.nan], type_sort[:-1]))
    crossings_up = up_down_sort[(prev_type == -1) & (type_sort == 1)]
    crossings_down = up_down_sort[(prev_type == 1) & (type_sort == -1)]

    crossings = {'Up': crossings_up, 'Dwn': crossings_down}

    if len(crossings_up) == 1 and len(crossings_down) == 1:
        if crossings_up < crossings_down:
            crossings['Area'] = np.array([[crossings_up[0], crossings_down[0]]])
        else:
            crossings['Area'] = np.array([[1, crossings_up[0]], [crossings_down[0], len_signal - 1]]).T
    elif len(crossings_up) + len(crossings_down) == 0:
        crossings['Area'] = np.array([]).reshape(0, 0)
    elif len(crossings_up) == 0:
        crossings['Area'] = np.array([1, crossings_down[0]])
    elif len(crossings_down) == 0:
        crossings['Area'] = np.array([crossings_up[0], len_signal - 1])
    else:
        if crossings_up[-1] > crossings_down[-1] and crossings_up[0] > crossings_down[0]:
            crossings['Dwn'] = np.append(crossings_down, len_signal - 1)
            crossings['Up'] = np.append([1], crossings_up)
        elif crossings_up[-1] > crossings_down[-1] and crossings_up[0] < crossings_down[0]:
            crossings['Dwn'] = np.append(crossings_down, len_signal - 1)
        elif crossings_up[-1] < crossings_down[-1] and crossings_up[0] > crossings_down[0]:
            crossings['Up'] = np.append([1], crossings_up)
        crossings['Area'] = np.vstack((crossings['Up'], crossings['Dwn'])).T

    return crossings


def _normalize_crossing_area(area):
    """Return trigger areas as an ``(n_events, 2)`` integer array."""
    area = np.asarray(area)
    if area.size == 0:
        return np.empty((0, 2), dtype=int)

    area = np.atleast_2d(area)
    if area.shape[1] != 2 and area.shape[0] == 2:
        area = area.T
    if area.shape[1] != 2:
        return np.empty((0, 2), dtype=int)

    area = area.astype(int, copy=False)
    area = area[area[:, 1] > area[:, 0]]
    return area


# =====================================================================
# Stage 4: FrequencyClusterer
# =====================================================================

class FrequencyClusterer:
    def __init__(self, coeff, freqs, config: ClusteringConfig = None):
        self.coeff = coeff
        self.freqs = freqs

        # use provided config or defaults
        self.config = config or ClusteringConfig()


    def run(self):
        # Normalize
        if self.config.standardize:
            mean = np.mean(self.coeff, axis=1, keepdims=True)
            std = np.std(self.coeff, axis=1, ddof=0, keepdims=True)
            cwt_coeff = (self.coeff - mean) / std
        else:
            cwt_coeff = self.coeff

        # PCA
        pca = PCA(n_components=self.config.n_components)
        pca_coeff = pca.fit_transform(np.abs(cwt_coeff))
        explained = pca.explained_variance_ratio_

        # Clustering
        Z = linkage(pca_coeff[:, :self.config.n_comp_clu], method=self.config.method_linkage)
        if self.config.method_fclust == 'maxclust':
            labels = fcluster(Z, self.config.n_clusters, criterion='maxclust')
            sublabels = fcluster(Z, self.config.n_subclusters, criterion='maxclust')
        elif self.config.method_fclust == 'distance':
            threshold = self.config.thres_dist * max(Z[:, 2])
            labels = fcluster(Z, t=threshold, criterion='distance')
            sublabels = fcluster(Z, t=threshold / 2, criterion='distance')
        else:
            raise ValueError(f"Unsupported method_fclust: {self.config.method_fclust}")

        return {
            "pca_coeff": pca_coeff,
            "explained": explained,
            "Z": Z,
            "labels": labels,
            "frequencies": {i: self.freqs[labels == i] for i in range(1, self.config.n_clusters + 1)},
            "sublabels": sublabels,
            "subfreqs": {i: self.freqs[sublabels == i] for i in range(1, self.config.n_subclusters + 1)}
        }


# =====================================================================
# Stage 5: WaveletReducer (adapted from reducer.py)
# =====================================================================

class WaveletReducer:
    def __init__(self, cluster_result, coeff, freqs, scales, config: cwtReducerConfig = None):
        self.cluster_result = cluster_result
        self.coeff = coeff
        self.freqs = freqs
        self.scales = scales

        # use provided config or defaults
        self.config = config or cwtReducerConfig()

    def reduce_and_threshold(self):
        event_onsets = {}
        reduced_coeff, reduced_freq, reduced_scale = [], [], []
        selected_clu = []

        for i, freq_id in enumerate(np.unique(self.cluster_result[self.config.cluster_label])):
            freq_range = self.cluster_result[self.config.cluster_freq][freq_id]
            mask = np.isin(self.freqs, freq_range)
            if not np.any(mask):
                continue

            avg_coeff = np.mean(self.coeff[mask, :], axis=0)
            if not self.config.complex_bands:
                avg_coeff = avg_coeff.real
            avg_scale = np.mean(self.scales[mask], axis=0)
            min_f, max_f = min(freq_range), max(freq_range)
            max_coeff = np.max(avg_coeff.real)

            # smoothed bands are real either way, the fit runs on the real part
            if max_f <= 15:
                avg_coeff = savgol_filter(avg_coeff.real, window_length=41, polyorder=1)
                max_coeff_sm = np.max(avg_coeff)
                if max_coeff_sm != 0:
                    avg_coeff = avg_coeff * (max_coeff / max_coeff_sm)
            elif 15 < max_f <= 50:
                avg_coeff = savgol_filter(avg_coeff.real, window_length=21, polyorder=1)
                max_coeff_sm = np.max(avg_coeff)
                if max_coeff_sm != 0:
                    avg_coeff = avg_coeff * (max_coeff / max_coeff_sm)
            elif 50 < max_f <= 75:
                avg_coeff = savgol_filter(avg_coeff.real, window_length=11, polyorder=1)
                max_coeff_sm = np.max(avg_coeff)
                if max_coeff_sm != 0:
                    avg_coeff = avg_coeff * (max_coeff / max_coeff_sm)

            # if max_f > self.config.cutoff_freq[1] or max_f <= self.config.cutoff_freq[0]:
            #     print(max_f)
            #     continue

            up_thresh, down_thresh = (
                (np.std(avg_coeff) * self.config.slow_upthres, np.std(avg_coeff)) if max_f < 30
                else (np.std(avg_coeff) * self.config.fast_upthres, 0)
            )

            crossings = trigger_detect(avg_coeff, up_thresh, down_thresh)
            area = _normalize_crossing_area(crossings["Area"])
            if area.size == 0:
                continue

            iei = area[:, 1] - area[:, 0]
            event_onsets[f"clu{i}_{np.around(max_f, 1)}Hz"] = {
                "crossings": area, "iei": iei, "num_event": len(iei)
            }

            reduced_coeff.append(avg_coeff)
            reduced_freq.append([min_f, max_f])
            reduced_scale.append(avg_scale)
            selected_clu.append(i)

        return {
            "reduced_coeff": np.array(reduced_coeff),
            "reduced_freqs": np.array(reduced_freq),
            "reduced_scale": np.array(reduced_scale),
            "event_onsets": event_onsets,
            "selected_clusters": np.array(selected_clu)
        }


# =====================================================================
# Stage 6: AdaptiveThreshold (adapted from thresholding.py)
# =====================================================================

class AdaptiveThreshold:
    def __init__(self, reduced_cwt, config: thresConfig = None):
        self.reduced = reduced_cwt
        self.n_clu, self.n_frame = np.shape(self.reduced["reduced_coeff"])

        self.config = config or thresConfig()

    def run(self):
        event_onsets = self.reduced["event_onsets"].copy()
        # selected_clusters = self.reduced['selected_clusters']
        selected_freqs = self.reduced['reduced_freqs'][:, 1].copy()

        num_labels = len(event_onsets)
        label_matrix = np.zeros((num_labels, self.n_frame))

        if self.config.thres_type == "soft":
            # Initial attenuation level by frequency
            low, mid, high, top = self.config.soft_levels
            for i, freq in enumerate(selected_freqs):
                if freq < 5:
                    label_matrix[i, :] = low
                elif 5 <= freq < 30:
                    label_matrix[i, :] = mid
                elif 30 <= freq < 80:
                    label_matrix[i, :] = high
                else:
                    label_matrix[i, :] = top
        elif self.config.thres_type == "hard":
            label_matrix = label_matrix + self.config.hard_floor

        # Fill in detected event regions
        for i, (event_id, data) in enumerate(event_onsets.items()):
            crossings = data["crossings"]
            median_iei = np.median(data["iei"])

            if selected_freqs[i] <= 20:
                extra_bin = int(median_iei * 2.0)
            elif 20 < selected_freqs[i] <= 50:
                extra_bin = int(median_iei * 1.5)
            else:
                extra_bin = int(median_iei * 0.5)

            for st, ed in crossings:
                st_bin = max(st - extra_bin, 0)
                ed_bin = min(ed + extra_bin, self.n_frame)
                label_matrix[i, st_bin:ed_bin] = 1

        # Scale and smooth
        recon_data = self.reduced["reduced_coeff"][:, :].copy()
        scale_factors = np.sqrt(self.reduced["reduced_scale"][:, np.newaxis])
        recon_scaled = recon_data / scale_factors

        sigmas = self.config.soft_sigma if self.config.thres_type == "soft" else self.config.hard_sigma
        for i, freq in enumerate(selected_freqs):
            sigma = sigmas[0] if freq <= 120 else sigmas[1]
            smooth_label = gaussian_filter1d(label_matrix[i], sigma=sigma)
            recon_data[i, :] *= smooth_label
            recon_scaled[i, :] *= smooth_label

        return {
            "clu_label": label_matrix,
            "all_scaled": recon_scaled,
            "all": recon_data,
            "rescaled_signal": np.sum(recon_scaled, axis=0, keepdims=True),
            "scaling_factor": scale_factors,
            "selected_freq": self.reduced["reduced_freqs"][:, :]
        }


def fir_lowpass(signal, fs, cutoff_hz, window_ms=2000.0, *, odd_taps=True):
    """Zero-phase FIR low-pass (Hamming window, ``filtfilt``).

    Parameters
    ----------
    signal : array-like, shape (n_frame,)
    fs : float
        Sampling rate in Hz.
    cutoff_hz : float
    window_ms : float, default 2000
        Window length in milliseconds; the tap count is
        ``int(fs * window_ms / 1000)``, capped for short signals.
    odd_taps : bool, default True
        Trim an even tap count to odd (a symmetric type-I filter). The
        spatial JEDI archive was made with the even count (``False``).

    Returns
    -------
    np.ndarray
        The low-pass trace; a copy of the input when it is too short to filter.
    """
    signal = np.asarray(signal).flatten()
    if signal.size < 8:
        return signal.copy()

    nyquist = fs / 2
    cutoff = min(cutoff_hz, nyquist * 0.99)
    if cutoff <= 0:
        return np.zeros_like(signal)

    taps = int(fs * window_ms / 1000)
    taps = max(3, taps)
    max_taps = (signal.size - 2) // 3
    if max_taps < 3:
        return signal.copy()
    taps = min(taps, max_taps)
    if odd_taps and taps % 2 == 0:
        taps -= 1
    if taps < 3:
        return signal.copy()

    fir_coefficients = firwin(taps, cutoff=cutoff, window="hamming", pass_zero=True, fs=fs)
    return filtfilt(fir_coefficients, 1.0, signal)


# =====================================================================
# Denoiser: end-to-end wrapper for a single 1D ΔF/F trace
# =====================================================================

class Denoiser:
    """Single-trace denoising pipeline for z-scored delta F/F signals.

    ``Denoiser`` denoises one fluorescence trace by computing a continuous
    wavelet transform, clustering frequency bands, reducing each selected
    cluster to a representative coefficient trace, applying an adaptive event
    mask, and restoring the low-frequency baseline with a zero-phase FIR
    filter.

    Parameters
    ----------
    fs : float
        Sampling rate of the recording in Hz.
    wavelet : str, default "cmor0.5-1.0"
        Wavelet identifier passed to :func:`pywt.cwt`.
    freq_scales : array-like of float, optional
        CWT scales passed to :func:`pywt.cwt`. If omitted, 100 log-spaced
        scales from 1 to 1000 are used.
    lp_cutoff : float, default 1.0
        Low-pass cutoff in Hz for the baseline added back at the end.
    fir_window_ms : float, default 2000
        FIR Hamming-window length in milliseconds.
    cfg_clust : ClusteringConfig, optional
        Frequency clustering configuration. Defaults to
        :class:`ClusteringConfig`.
    cfg_reducer : cwtReducerConfig, optional
        Wavelet reduction and event detection configuration. Defaults to
        :class:`cwtReducerConfig`.
    cfg_thres : thresConfig, optional
        Adaptive thresholding configuration. Defaults to
        ``thresConfig(thres_type="soft")``.
    fir_odd_taps : bool, default True
        Trim the FIR baseline filter to an odd tap count; see
        :func:`fir_lowpass`.

    Attributes
    ----------
    coeff_ : np.ndarray or None
        CWT coefficients from the most recent run.
    freqs_ : np.ndarray or None
        Frequencies returned by the CWT for the most recent run.
    cluster_result_ : dict or None
        Frequency clustering output from the most recent run.
    reduced_ : dict or None
        Reduced wavelet coefficients and event windows from the most recent
        run.
    threshold_result_ : dict or None
        Adaptive thresholding output from the most recent run.
    rescaled_signal_ : np.ndarray or None
        The masked wavelet sum of the most recent run, shape ``(n_frame,)``:
        the denoised trace without its baseline. Complex when
        ``cfg_reducer.complex_bands`` is set.
    lp_dfof_ : np.ndarray or None
        Low-pass baseline from the most recent run.
    denoised_ : np.ndarray or None
        Final denoised trace from the most recent run.
    event_indices_ : np.ndarray or None
        Sorted event start indices from the most recent run.
    event_ranges_ : dict or None
        Per-cluster event windows from the most recent run.
    timing_ : dict or None
        Wall seconds of each stage of the most recent run, in the order they
        ran: ``cwt``, ``cluster``, ``reduce``, ``mask``, ``baseline``.
    """

    def __init__(
        self,
        fs,
        wavelet='cmor0.5-1.0',
        freq_scales=None,
        lp_cutoff=1.0,
        fir_window_ms=2000,
        cfg_clust=None,
        cfg_reducer=None,
        cfg_thres=None,
        fir_odd_taps=True,
    ):
        self.fs = fs
        self.wavelet = wavelet
        self.freq_scales = (
            np.asarray(freq_scales) if freq_scales is not None
            else np.logspace(np.log10(1), np.log10(1000), num=100)
        )
        self.lp_cutoff = lp_cutoff
        self.fir_window_ms = fir_window_ms
        self.fir_odd_taps = fir_odd_taps
        self.cfg_clust = cfg_clust or ClusteringConfig()
        self.cfg_reducer = cfg_reducer or cwtReducerConfig()
        self.cfg_thres = cfg_thres or thresConfig(thres_type="soft")

        # populated by run() — exposed for downstream inspection / plotting
        self.coeff_ = None
        self.freqs_ = None
        self.cluster_result_ = None
        self.reduced_ = None
        self.threshold_result_ = None
        self.rescaled_signal_ = None
        self.lp_dfof_ = None
        self.denoised_ = None
        self.event_indices_ = None
        self.event_ranges_ = None
        self.timing_ = None

    @classmethod
    def upstream(cls, fs, **overrides):
        """The settings the spatial JEDI archive was made with.

        100 log-spaced scales 1...1000, PCA 30 / 10 used, 5 clusters / 10
        bands, open levels 2.0 / 2.5 SD, soft masks with the lab's attenuation
        (0.7 / 0.5 / 0.2 / 0.01), complex band traces above 75 Hz, and an even
        FIR tap count for the 1 Hz baseline. ``run`` then reproduces
        ``denoised_trace_components.pkl`` (``rescaled_signal_`` and
        ``lp_dfof_``) to float precision when ``fs`` is the exact frame rate.
        Keyword arguments override constructor parameters.
        """
        settings = dict(
            cfg_clust=ClusteringConfig(),
            cfg_reducer=cwtReducerConfig(complex_bands=True),
            cfg_thres=thresConfig(thres_type="soft", soft_levels=UPSTREAM_SOFT_LEVELS),
            fir_odd_taps=False,
        )
        settings.update(overrides)
        return cls(fs, **settings)

    def describe(self):
        """The settings as plain data, for provenance files."""
        from dataclasses import asdict

        scales = np.asarray(self.freq_scales, dtype=float)
        return {
            "fs": float(self.fs),
            "wavelet": self.wavelet,
            "freq_scales": {
                "min": float(scales.min()),
                "max": float(scales.max()),
                "num": int(scales.size),
                "spacing": "log",
            },
            "lp_cutoff": float(self.lp_cutoff),
            "fir_window_ms": float(self.fir_window_ms),
            "fir_odd_taps": bool(self.fir_odd_taps),
            "clustering": asdict(self.cfg_clust),
            "reducer": asdict(self.cfg_reducer),
            "threshold": asdict(self.cfg_thres),
        }

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _cwt(self, dfof):
        """Compute the continuous wavelet transform.

        Parameters
        ----------
        dfof : np.ndarray
            Flattened z-scored delta F/F trace.

        Returns
        -------
        coefficients : np.ndarray
            Wavelet coefficient matrix with one row per scale.
        frequencies : np.ndarray
            Frequencies corresponding to ``coefficients``.
        """
        coefficients, frequencies = pywt.cwt(
            dfof,
            scales=self.freq_scales,
            wavelet=self.wavelet,
            sampling_period=1 / self.fs,
        )
        return coefficients, frequencies

    def _cluster(self, coeff, freqs):
        """Cluster CWT frequency rows.

        Parameters
        ----------
        coeff : np.ndarray
            CWT coefficient matrix.
        freqs : np.ndarray
            Frequencies corresponding to rows in ``coeff``.

        Returns
        -------
        dict
            Frequency clustering results, including labels, sublabels,
            linkage output, and PCA scores.
        """
        return FrequencyClusterer(coeff, freqs, config=self.cfg_clust).run()

    def _reduce(self, cluster_result, coeff, freqs):
        """Reduce clustered CWT rows and detect event windows.

        Parameters
        ----------
        cluster_result : dict
            Output from :meth:`_cluster`.
        coeff : np.ndarray
            CWT coefficient matrix.
        freqs : np.ndarray
            Frequencies corresponding to rows in ``coeff``.

        Returns
        -------
        dict
            Reduced coefficient traces, selected frequency ranges, detected
            event windows, and selected cluster indices.
        """
        return WaveletReducer(
            cluster_result,
            coeff,
            freqs,
            self.freq_scales,
            config=self.cfg_reducer,
        ).reduce_and_threshold()

    def _adaptive_threshold(self, reduced):
        """Apply adaptive event masking and combine selected clusters.

        Parameters
        ----------
        reduced : dict
            Output from :meth:`_reduce`.

        Returns
        -------
        dict
            Thresholded and scale-corrected reconstruction data.
        """
        return AdaptiveThreshold(reduced, config=self.cfg_thres).run()

    def _fir_lowpass(self, signal):
        """Estimate the low-frequency baseline with a zero-phase FIR filter.

        Parameters
        ----------
        signal : np.ndarray
            Input trace.

        Returns
        -------
        np.ndarray
            Low-pass filtered baseline trace.
        """
        return fir_lowpass(
            signal, self.fs, self.lp_cutoff, self.fir_window_ms, odd_taps=self.fir_odd_taps,
        )

    @staticmethod
    def _collect_event_indices(event_onsets):
        """Collect event starts and per-cluster event windows.

        Parameters
        ----------
        event_onsets : dict
            Event windows returned by :class:`WaveletReducer`.

        Returns
        -------
        starts : np.ndarray, shape (n_events,)
            Sorted frame indices of event start crossings, across all clusters.
        ranges : dict
            Per-cluster ``(n_events, 2)`` arrays with start and end frame
            indices.
        """
        all_starts = []
        ranges = {}
        for clu_id, data in event_onsets.items():
            crossings = data["crossings"]
            if crossings.size == 0:
                continue
            ranges[clu_id] = crossings
            all_starts.extend(crossings[:, 0].tolist())
        starts = np.sort(np.array(all_starts, dtype=int))
        return starts, ranges

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, dfof, cwt=None):
        """Denoise one z-scored delta F/F trace.

        The pipeline performs CWT decomposition, frequency clustering,
        cluster-level reduction, adaptive event masking, low-pass baseline
        estimation, and event-index collection. Intermediate outputs are stored
        on the instance attributes ending in an underscore, with each stage's
        wall seconds on ``timing_``.

        Parameters
        ----------
        dfof : array-like, shape (n_frame,)
            One-dimensional z-scored delta F/F trace.
        cwt : tuple of (coefficients, frequencies), optional
            A transform of ``dfof`` computed earlier (a saved ``cwts.h5``
            slice, say) to reuse instead of running :func:`pywt.cwt`, by far
            the slowest stage.

        Returns
        -------
        denoised : np.ndarray, shape (n_frame,)
            Denoised trace, computed as wavelet event reconstruction plus the
            low-pass baseline.
        event_indices : np.ndarray, shape (n_events,)
            Sorted frame indices of detected event starts (across all clusters).
        """
        dfof = np.asarray(dfof).flatten()
        self.timing_ = {}

        # 3 — CWT
        started = time.perf_counter()
        if cwt is None:
            coeff, freqs = self._cwt(dfof)
        else:
            coeff, freqs = cwt
            coeff = np.asarray(coeff)
            freqs = np.asarray(freqs, dtype=float)
            if coeff.shape != (self.freq_scales.size, dfof.size):
                raise ValueError(
                    f"cwt coefficients {coeff.shape} do not match "
                    f"{self.freq_scales.size} scales x {dfof.size} samples"
                )
        self.coeff_, self.freqs_ = coeff, freqs
        self.timing_["cwt"] = time.perf_counter() - started

        # 4 — frequency clustering
        started = time.perf_counter()
        cluster_result = self._cluster(coeff, freqs)
        self.cluster_result_ = cluster_result
        self.timing_["cluster"] = time.perf_counter() - started

        # 5 — reduce + threshold
        started = time.perf_counter()
        reduced = self._reduce(cluster_result, coeff, freqs)
        self.reduced_ = reduced
        self.timing_["reduce"] = time.perf_counter() - started

        # 6 — adaptive mask + sum
        started = time.perf_counter()
        thres_result = self._adaptive_threshold(reduced)
        self.threshold_result_ = thres_result
        self.timing_["mask"] = time.perf_counter() - started

        # 7 — FIR low-pass baseline + event reconstruction
        started = time.perf_counter()
        rescaled_signal = thres_result['rescaled_signal']    # (1, n_frame)
        self.rescaled_signal_ = rescaled_signal[0]
        lp_dfof = self._fir_lowpass(dfof)                    # (n_frame,)
        self.lp_dfof_ = lp_dfof
        denoised = rescaled_signal[0] + lp_dfof              # (n_frame,)
        self.denoised_ = denoised
        self.timing_["baseline"] = time.perf_counter() - started

        # event indices
        starts, ranges = self._collect_event_indices(reduced['event_onsets'])
        self.event_indices_ = starts
        self.event_ranges_ = ranges

        return denoised, starts
