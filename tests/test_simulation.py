import numpy as np

from vnoiser import Jedi3SubConfig, simulate_jedi3sub_trace
from vnoiser.denoiser import ClusteringConfig, Denoiser, cwtReducerConfig, thresConfig


def test_simulation_shape_and_events_are_valid():
    cfg = Jedi3SubConfig(
        fs_hz=1000.0,
        isolated_ap_rate_hz=5.0,
        fast_event_rate_hz=20.0,
        slow_event_rate_hz=2.0,
        burst_rate_hz=1.0,
        ipsp_rate_hz=4.0,
    )
    result = simulate_jedi3sub_trace(1.0, config=cfg, seed=3)

    assert len(result.t) == 1000
    assert result.voltage_mv.shape == result.dff_z.shape
    assert np.isfinite(result.voltage_mv).all()
    assert np.isfinite(result.dff_clean).all()
    assert np.isfinite(result.dff_observed).all()
    assert np.isfinite(result.dff_z).all()
    assert abs(np.mean(result.dff_z)) < 1e-10
    assert abs(np.std(result.dff_z) - 1.0) < 1e-10
    assert result.metadata["polarity"].startswith("bright-to-dim")

    for key, values in result.events.items():
        values = np.asarray(values)
        if key.endswith("_indices"):
            assert ((0 <= values) & (values < len(result.t))).all()
        elif key.endswith("_times_s"):
            assert ((0 <= values) & (values < 1.0)).all()


def test_depolarizing_events_are_negative_fluorescence_deflections():
    cfg = Jedi3SubConfig(
        fs_hz=10000.0,
        isolated_ap_rate_hz=0.0,
        fast_event_rate_hz=0.0,
        slow_event_rate_hz=0.0,
        burst_rate_hz=3.0,
        ipsp_rate_hz=0.0,
        ripple_rate_hz=0.0,
        photon_rate_hz=1.0e9,
        read_noise_photons=0.0,
        electronic_noise_dff=0.0,
        scanline_gain_sd=0.0,
        motion_artifact_dff=0.0,
        bleach_fraction=0.0,
    )
    result = simulate_jedi3sub_trace(1.0, config=cfg, seed=5)

    assert len(result.events["burst_ap_indices"]) > 0
    first_ap = result.events["burst_ap_indices"][0]
    post_window = result.dff_clean[first_ap : first_ap + int(0.012 * cfg.fs_hz)]
    assert np.min(post_window) < -0.05


def test_simulation_is_deterministic_with_seed():
    cfg = Jedi3SubConfig(fs_hz=1000.0)
    first = simulate_jedi3sub_trace(0.5, config=cfg, seed=10)
    second = simulate_jedi3sub_trace(0.5, config=cfg, seed=10)

    np.testing.assert_allclose(first.voltage_mv, second.voltage_mv)
    np.testing.assert_allclose(first.dff_z, second.dff_z)
    np.testing.assert_array_equal(first.events["ap_indices"], second.events["ap_indices"])


def test_denoiser_runs_on_short_simulation():
    cfg = Jedi3SubConfig(
        fs_hz=1000.0,
        isolated_ap_rate_hz=8.0,
        fast_event_rate_hz=12.0,
        slow_event_rate_hz=2.0,
        burst_rate_hz=1.0,
        ipsp_rate_hz=4.0,
        photon_rate_hz=1.5e6,
    )
    result = simulate_jedi3sub_trace(0.8, config=cfg, seed=4)
    denoiser_input = -result.dff_z
    model = Denoiser(
        fs=cfg.fs_hz,
        freq_scales=np.logspace(np.log10(1), np.log10(250), num=40),
        lp_cutoff=12.0,
        fir_window_ms=80.0,
        cfg_clust=ClusteringConfig(n_components=8, n_comp_clu=5, n_clusters=4, n_subclusters=5),
        cfg_reducer=cwtReducerConfig(slow_upthres=1.5, fast_upthres=2.0),
        cfg_thres=thresConfig(thres_type="soft"),
    )

    denoised, starts = model.run(denoiser_input)

    assert denoised.shape == denoiser_input.shape
    assert starts.ndim == 1
    assert np.isfinite(denoised).all()
