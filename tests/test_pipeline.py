"""Stages 0-5 and the PF writer: synthetic checks, plus parity with the
spatial JEDI archive when ``stan112_expt12`` is reachable."""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from vnoiser import (
    Denoiser,
    DfofConfig,
    PfWriter,
    ScanTraces,
    SpatialJediDataset,
    SpikeDetectConfig,
    detect_peaks,
    domain_zscore,
    final_trace,
    load_scan_rois,
    load_vi,
    process_domain,
    read_pf,
    run_from_vi,
    run_pipeline,
)
from vnoiser.pf import CWT_FILE, DFOF_FILE, PARAMS_FILE, PEAKS_FILE, PROVENANCE_FILE, TRACES_FILE

ARCHIVE = Path(os.environ.get("VNOISER_ARCHIVE", "X:/data/asako/stan112/stan112_expt12"))
PF = ARCHIVE / "PF"
# the archive's frame rate: TStepInMs = 0.93 in the .mesc
FS_EXACT = 1000.0 / 0.93
archive = pytest.mark.skipif(not (PF / TRACES_FILE).exists(), reason="archive not reachable")


def test_domain_zscore_weights_rois_by_pixels_and_replaces_the_startup():
    rng = np.random.default_rng(1)
    n = 4000
    pixels = {0: rng.normal(1000, 5, (n, 10)), 1: rng.normal(1200, 5, (n, 20))}
    traces = {r: p.mean(axis=1) for r, p in pixels.items()}
    weights = {0: 10.0, 1: 20.0}
    cfg = DfofConfig(sigma_dfof=100, sigma_baseline=300, n_startup=50)
    out = domain_zscore(traces, {"All_domains": [0, 1], "soma1": [0, 1]}, weights, cfg)

    assert out.names == ["soma1"]
    # the pixel mean of the concatenated rois, as the archive computed it
    concat = np.concatenate([pixels[0], pixels[1]], axis=1).mean(axis=1)
    expected = (concat - gaussian_filter1d(concat, 100)) / gaussian_filter1d(concat, 100)
    expected[:50] = expected[50:].mean()
    assert np.allclose(out.dfof_raw[0], expected)
    assert np.allclose(out.dfof_raw[0, :50], out.dfof_raw[0, 0])
    z_expected = -(expected - gaussian_filter1d(expected, 300)) / expected.std()
    assert np.allclose(out.z[0], z_expected)

    unweighted = domain_zscore(traces, {"soma1": [0, 1]}, None, cfg)
    assert not np.allclose(unweighted.dfof_raw[0], expected)


def test_detect_peaks_recovers_synthetic_spikes():
    fs = 1000.0
    rng = np.random.default_rng(2)
    n = int(10 * fs)
    # a denoised trace is smooth between events
    y = gaussian_filter1d(rng.normal(0, 0.3, n), 4)
    kernel = 1.5 * np.exp(-np.arange(int(0.012 * fs)) / (0.004 * fs))
    truth = np.array([1500, 2600, 4000, 5200, 7100, 8800])
    for i in truth:
        y[i:i + kernel.size] += kernel
    found = detect_peaks(y, fs, SpikeDetectConfig())
    assert found.size == truth.size
    assert np.all(np.abs(found - truth) <= 2)

    cfg = SpikeDetectConfig.from_param_pickle({"thres_bp_sd": 3, "thres_amp_sd": 2, "bp": [5, 400], "duration_thres": 10})
    assert cfg.bp == (5.0, 400.0) and cfg.duration_thres_ms == 10.0
    assert cfg.to_param_pickle() == {"thres_bp_sd": 3.0, "thres_amp_sd": 2.0, "bp": [5.0, 400.0], "duration_thres": 10.0}
    assert detect_peaks(np.zeros(10), fs).size == 0


def test_final_trace_is_the_real_part_plus_the_baseline_start():
    rescaled = np.array([1 + 1j, 2 - 1j, 0.5 + 0j])
    lp = np.array([0.25, 0.5, 0.75])
    assert np.allclose(final_trace(rescaled, lp), [1.25, 2.25, 0.75])


def test_upstream_preset_and_precomputed_cwt():
    model = Denoiser.upstream(1000.0)
    info = model.describe()
    assert info["threshold"]["thres_type"] == "soft"
    assert tuple(info["threshold"]["soft_levels"]) == (0.7, 0.5, 0.2, 0.01)
    assert info["reducer"]["complex_bands"] is True
    assert info["fir_odd_taps"] is False
    assert info["freq_scales"] == {"min": 1.0, "max": 1000.0, "num": 100, "spacing": "log"}

    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, 3000)
    z[1000:1010] += 6
    z[2000:2010] += 6
    denoised, _ = model.run(z)
    coeff, freqs = model.coeff_, model.freqs_
    again = Denoiser.upstream(1000.0)
    denoised2, _ = again.run(z, cwt=(coeff, freqs))
    assert np.allclose(denoised, denoised2)
    assert np.iscomplexobj(again.rescaled_signal_)
    with pytest.raises(ValueError):
        Denoiser.upstream(1000.0).run(z, cwt=(coeff[:, :100], freqs))


def test_pipeline_writes_a_pf_folder_the_dataset_opens(tmp_path):
    # three line rois (10, 20, 10 px) of a jedi-like recording: bright baseline, slow drift, negative-going spikes
    fs = 1000.0
    rng = np.random.default_rng(0)
    n = int(fs * 8.0)
    t = np.arange(n) / fs
    spikes = np.zeros(n)
    kernel = np.exp(-np.arange(int(0.008 * fs)) / (0.003 * fs))
    for i in rng.integers(int(1.5 * fs), n - int(0.1 * fs), size=12):
        spikes[i:i + kernel.size] += kernel[: n - i]
    base = 1700 + 40 * np.sin(2 * np.pi * 0.2 * t)
    traces, weights = {}, {}
    for roi, width in enumerate((10, 20, 10)):
        pixels = base[:, None] * (1 - 0.06 * spikes[:, None]) + rng.normal(0, 25, (n, width))
        traces[roi] = pixels.mean(axis=1)
        weights[roi] = float(width)
    scan = ScanTraces("10", fs, traces, weights, comment="soma,api1")

    experiment = tmp_path / "stan9" / "stan9_expt1"
    domains = {"soma1": [0, 1], "apical1": [2]}
    paths = run_pipeline(
        [scan], experiment / "PF", domains=domains, first_env=["10"], save_cwt=True,
        provenance={"source": {"mesc": "synthetic"}},
    )
    for name in (TRACES_FILE, DFOF_FILE, CWT_FILE, PEAKS_FILE, PARAMS_FILE, PROVENANCE_FILE):
        assert name in paths

    files = read_pf(experiment / "PF")
    assert files.scan_ids == ["10"]
    assert files.fs == {"10": 1000}
    assert set(files.traces["10"]) == {"soma", "apical1"}
    assert files.domains["All_domains"] == [0, 1, 2]
    assert files.domains["soma1"] == [0, 1]
    assert files.rois["roi_list"] == {"10": [0, 1, 2]}
    assert files.rois["scanID_1st_env"] == ["10"]
    assert files.params["bp"] == [2.0, 400.0]
    assert files.provenance["source"] == {"mesc": "synthetic"}
    assert files.provenance["denoiser"]["threshold"]["thres_type"] == "soft"
    assert "numpy" in files.provenance["versions"]
    assert files.provenance["roi_weights"] == {"10": {"0": 10.0, "1": 20.0, "2": 10.0}}

    with h5py.File(experiment / "PF" / DFOF_FILE) as f:
        assert f["10/dfof_raw"].shape == (2, n)
        assert f["10/dfof_zscore"].shape == (2, n)
        z_soma = f["10/dfof_zscore"][0]
    with h5py.File(experiment / "PF" / CWT_FILE) as f:
        assert f["10/cwt_dfof"].shape == (100, n, 2)
        assert f["10/cwt_dfof"].dtype == np.complex64
        assert f["10/freq_scale"][0] == pytest.approx(1.0)
        assert f["10/cwt_freq"][0, 0] == pytest.approx(1000.0)

    # the same domain processed on its own gives the written trace
    single = process_domain(z_soma, 1000.0)
    assert np.allclose(single.trace, files.traces["10"]["soma"])
    assert np.array_equal(single.peaks, files.peaks["10"]["soma"])
    # the z-score flips the negative-going spikes up and the detector finds most of them
    truth = np.flatnonzero(np.diff(spikes > 0.5, prepend=0) == 1)
    assert np.isin(files.peaks["10"]["soma"], truth[:, None] + np.arange(-2, 12)).mean() > 0.6

    # the curation dataset lists the pipeline's domain names and finds the renamed soma trace
    dataset = SpatialJediDataset(experiment)
    ids = [rid for _, rid in dataset.recording_options()]
    assert ids == ["stan9/stan9_expt1/scan=10/domain=soma1", "stan9/stan9_expt1/scan=10/domain=apical1"]
    sample = dataset.load(ids[0], load_events=True)
    assert sample.fs_hz == 1000.0
    assert sample.metadata["pre_denoised"] is True
    assert np.allclose(sample.trace, files.traces["10"]["soma"])
    assert sample.events_ap_indices.size == files.peaks["10"]["soma"].size

    with pytest.raises(FileExistsError):
        PfWriter(experiment / "PF", domains=domains, scan_ids=["10"])
    again = run_pipeline([scan], experiment / "PF", domains=domains, overwrite=True, detect=False)
    assert PEAKS_FILE not in again and PARAMS_FILE not in again


def test_writer_checks_its_inputs(tmp_path):
    writer = PfWriter(tmp_path / "PF", domains={"soma1": [0]}, scan_ids=["1"])
    with pytest.raises(KeyError):
        writer.add_scan("2", 1000.0, np.zeros((1, 10)), np.zeros((1, 10)))
    with pytest.raises(ValueError):
        writer.add_scan("1", 1000.0, np.zeros((2, 10)), np.zeros((2, 10)))
    writer.add_scan("1", 1000.0, np.zeros((1, 10)), np.zeros((1, 10)))
    with pytest.raises(ValueError):
        writer.finish()


@archive
def test_stage_0_1_from_the_packaged_pickle_matches_test_h5():
    vi_path = next(ARCHIVE.glob("VI_*.pkl"))
    animal, experiment, scans = load_vi(vi_path)
    assert (animal, experiment) == ("stan112", "2")
    rois = load_scan_rois(PF / "scanIDs_ROIs.pkl")
    assert rois["scan_ids"] == ["35", "38"]
    assert scans["35"].weights[1] == 20.0 and scans["35"].weights[0] == 10.0
    with h5py.File(PF / DFOF_FILE) as f:
        for scan_id in rois["scan_ids"]:
            out = domain_zscore(scans[scan_id].traces, rois["domains"], scans[scan_id].weights)
            assert np.allclose(out.dfof_raw, f[f"{scan_id}/dfof_raw"][:], atol=1e-12)
            assert np.allclose(out.z, f[f"{scan_id}/dfof_zscore"][:], atol=1e-9)


@archive
def test_stages_2_5_from_the_stored_cwt_match_the_archive():
    theirs = read_pf(PF, components=True)
    spike_cfg = SpikeDetectConfig.from_param_pickle(theirs.params)
    names = list(theirs.components["35"])
    exact_peaks = {"basal1", "basal3", "apical1", "apical4", "soma1"}
    with h5py.File(PF / DFOF_FILE) as f, h5py.File(PF / CWT_FILE) as cw:
        for row, name in enumerate(names):
            z = f["35/dfof_zscore"][row].astype(float)
            coeff = cw["35/cwt_dfof"][:, :, row]
            freqs = cw["35/cwt_freq"][:, row]
            model = Denoiser.upstream(FS_EXACT)
            model.run(z, cwt=(coeff, freqs))
            stored = theirs.components["35"][name]
            assert np.abs(model.rescaled_signal_ - np.asarray(stored["rescaled_signal"]).ravel()).max() < 1e-5, name
            assert np.abs(model.lp_dfof_ - np.asarray(stored["lp_FIR1Hz"])).max() < 1e-9, name

            result = process_domain(
                z, FS_EXACT, denoiser=Denoiser.upstream(FS_EXACT), spike_cfg=spike_cfg, cwt=(coeff, freqs),
            )
            assert np.abs(result.lp_fir100hz - np.asarray(stored["lp_FIR100Hz"])).max() < 1e-9, name
            assert np.abs(result.envelope_lp - np.asarray(stored["envelope_lp"])).max() < 1e-9, name
            final_name = "soma" if name == "soma1" else name
            diff = np.abs(result.trace - theirs.traces["35"][final_name])
            # the archive's final trace has a handful of patched samples
            assert np.median(diff) < 1e-6 and (diff < 1e-5).mean() > 0.999, name
            ref = np.sort(theirs.peaks["35"][final_name])
            if name in exact_peaks:
                assert np.array_equal(result.peaks, ref), name
            else:
                assert np.isin(ref, result.peaks).all(), name
                assert result.peaks.size - ref.size <= 2, name


@archive
@pytest.mark.slow
def test_end_to_end_scan_35_from_the_packaged_pickle(tmp_path):
    rois = load_scan_rois(PF / "scanIDs_ROIs.pkl")
    theirs = read_pf(PF)
    paths = run_from_vi(
        next(ARCHIVE.glob("VI_*.pkl")), tmp_path / "stan112" / "stan112_expt12" / "PF",
        domains=rois["domains"], scan_ids=["35"], first_env=["35"],
        spike_cfg=SpikeDetectConfig.from_param_pickle(theirs.params),
    )
    ours = read_pf(paths[TRACES_FILE].parent)
    for name, trace in ours.traces["35"].items():
        diff = np.abs(trace - theirs.traces["35"][name])
        assert np.median(diff) < 1e-6 and (diff < 1e-5).mean() > 0.999, name
    assert ours.fs == {"35": 1075}
