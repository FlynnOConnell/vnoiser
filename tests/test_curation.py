import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

pytest.importorskip("ipywidgets")
pytest.importorskip("plotly")

from vnoiser.curation import (
    DECISION_BUTTON_HEIGHT_PX,
    DECISION_BUTTON_WIDTH_PX,
    DECISION_ROW_HEIGHT_PX,
    EventCurationDashboard,
    candidate_pca_embedding,
    curation_dir_for,
    lowpass_trace,
    threshold_event_calls,
    threshold_slider_scale,
)


def _write_recording(path):
    fs_hz = 1000.0
    t = np.arange(3000, dtype=float) / fs_hz
    trace = 0.05 * np.sin(2 * np.pi * 4 * t)
    trace[1500] = 4.0
    savemat(
        path,
        {
            "CAttached": {
                "fluo_mean": trace,
                "fluo_time": t,
                "events_AP": np.array([1500]),
            }
        },
    )


def _install_fake_pipeline(monkeypatch, event_indices=(700, 1500, 2300)):
    event_indices = np.asarray(event_indices, dtype=int)

    def fake_pipeline(self, sample):
        self.recording = sample
        self.raw_trace = sample.trace.astype(float)
        self.denoised = np.zeros_like(self.raw_trace)
        self.denoised[event_indices] = np.arange(len(event_indices)) + 4.0
        self.denoiser_input = self.denoised.copy()
        self.raw_cluster_starts = event_indices.copy()
        self.pca_event_indices = np.array([123], dtype=int)
        self.pca_event_score_z = np.zeros_like(self.raw_trace)
        self.pca_threshold_z = 3.0
        self.pipeline_cache_status = "fake pipeline"
        self._finalize_pipeline_state()

    monkeypatch.setattr(EventCurationDashboard, "_run_or_load_pipeline", fake_pipeline)
    monkeypatch.setattr(
        "vnoiser.curation.lowpass_trace",
        lambda trace, fs_hz, cutoff_hz: np.asarray(trace, dtype=float).copy(),
    )


def _write_spatial_recording(root, trace=None, catalog_indices=(100, 200)):
    pf_dir = root / "stan1" / "stan1_expt1" / "PF"
    pf_dir.mkdir(parents=True)
    fs_hz = 1000.0
    t = np.arange(4000, dtype=float) / fs_hz
    if trace is None:
        trace = 0.1 * np.sin(2 * np.pi * 5 * t)
        trace[[800, 2000, 3200]] += [4.0, 7.0, 5.0]
    payloads = {
        "denoised_trace_scans.pkl": {"10": {"soma": trace}},
        "fs_scans.pkl": {"10": fs_hz},
        "scanIDs_ROIs.pkl": {
            "scanID_spatial": np.array([10]),
            "domain_ROInumber": {"soma": [0]},
        },
        "detected_events_peaks.pkl": {
            "10": {"soma": np.asarray(catalog_indices, dtype=int)}
        },
    }
    for filename, payload in payloads.items():
        with (pf_dir / filename).open("wb") as handle:
            pickle.dump(payload, handle)
    return pf_dir


def test_curation_dashboard_is_lazy_and_uses_data_local_curation_dir(tmp_path, monkeypatch):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)

    def fail_if_pipeline_runs(*args, **kwargs):
        raise AssertionError("pipeline should not run during dashboard construction")

    monkeypatch.setattr(EventCurationDashboard, "_run_or_load_pipeline", fail_if_pipeline_runs)

    dashboard = EventCurationDashboard(data_path=recording, mode="fast", auto_load=False)

    assert dashboard.recording is None
    assert dashboard.label_path == tmp_path / ".curation" / "fast_template_curation.json"
    assert curation_dir_for(recording) == tmp_path / ".curation"
    assert len(dashboard.recordings) == 1
    assert [button.description for button in dashboard.decision_buttons.children] == [
        "Yes",
        "No",
        "Clear",
    ]
    assert all(button.disabled for button in dashboard.decision_buttons.children)


def test_retired_seeded_mode_is_rejected(tmp_path):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)

    with pytest.raises(ValueError, match="manual.*fast.*slow"):
        EventCurationDashboard(data_path=recording, mode="seeded", auto_load=False)


def test_curation_labels_save_and_reload_from_data_local_json(tmp_path, monkeypatch):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)

    _install_fake_pipeline(monkeypatch, event_indices=(1500,))

    dashboard = EventCurationDashboard(data_path=recording, mode="manual", auto_load=False)
    dashboard._load_selected_recording(None)
    dashboard._set_label("yes")

    label_path = tmp_path / ".curation" / "manual_template_curation.json"
    payload = json.loads(label_path.read_text())
    assert payload["mode"] == "manual"
    assert len(payload["events"]) == 1
    assert next(iter(payload["events"].values()))["label"] == "yes"

    reloaded = EventCurationDashboard(data_path=recording, mode="manual", auto_load=False)
    assert reloaded.labels == dashboard.labels


@pytest.mark.parametrize("mode", ["fast", "slow"])
def test_seeded_auto_call_can_be_curated_to_no_in_data_local_json(
    tmp_path,
    monkeypatch,
    mode,
):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)

    _install_fake_pipeline(monkeypatch, event_indices=(1500,))

    dashboard = EventCurationDashboard(data_path=tmp_path, mode=mode, auto_load=False)
    dashboard._load_selected_recording(None)

    assert dashboard.label_path == (
        tmp_path / ".curation" / f"{mode}_template_curation.json"
    )
    assert dashboard._label_for_index(0) == "auto_yes"
    assert dashboard.labels == {}
    assert dashboard._initial_auto_call_for_index(0) == "pass"

    dashboard._set_label("no")

    assert dashboard.labels[dashboard.event_keys[0]] == "no"
    assert dashboard._label_for_index(0) == "no"
    assert len(dashboard.template_source) == 0

    payload = json.loads(dashboard.label_path.read_text())
    event = next(iter(payload["events"].values()))
    assert payload["data_path"] == str(tmp_path)
    assert event["label"] == "no"
    assert event["manual_label"] == "no"
    assert event["curation_state"] == "manual"
    assert event["initial_auto_call"] == "pass"
    assert event["initial_template_cosine"] == pytest.approx(1.0)
    assert event["auto_template_threshold"] == pytest.approx(0.8)
    assert event["was_seed_template_source"] is True

    reloaded = EventCurationDashboard(data_path=tmp_path, mode=mode, auto_load=False)
    reloaded._load_selected_recording(None)

    assert reloaded.labels == dashboard.labels
    assert reloaded._label_for_index(0) == "no"


@pytest.mark.parametrize("mode", ["manual", "fast", "slow"])
def test_dashboard_controls_have_non_shrinking_render_contract(tmp_path, mode):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)

    dashboard = EventCurationDashboard(data_path=recording, mode=mode, auto_load=False)

    expected_width = f"{DECISION_BUTTON_WIDTH_PX}px"
    expected_height = f"{DECISION_BUTTON_HEIGHT_PX}px"
    for button in dashboard.decision_buttons.children:
        assert button.layout.width == expected_width
        assert button.layout.min_width == expected_width
        assert button.layout.max_width == expected_width
        assert button.layout.height == expected_height
        assert button.layout.min_height == expected_height
        assert button.layout.max_height == expected_height
        assert button.layout.flex == f"0 0 {DECISION_BUTTON_WIDTH_PX}px"
        assert "vnoiser-decision-action" in button._dom_classes

    assert dashboard.decision_buttons.layout.height == f"{DECISION_ROW_HEIGHT_PX}px"
    assert dashboard.decision_buttons.layout.min_height == f"{DECISION_ROW_HEIGHT_PX}px"
    assert dashboard.decision_buttons.layout.flex_flow == "row nowrap"
    assert dashboard.decision_buttons.layout.overflow == "visible"
    assert "vnoiser-decision-row" in dashboard.decision_buttons._dom_classes
    assert "vnoiser-curation-dashboard" in dashboard.ui._dom_classes
    assert dashboard.threshold_slider.orientation == "vertical"
    assert dashboard.threshold_slider.layout.height == "190px"
    assert dashboard.threshold_slider.readout is True
    assert dashboard.threshold_box.children[0].value.endswith("A1. Threshold</div>")
    assert len(dashboard.threshold_box.children) == 2
    assert (
        f"min-height: {DECISION_BUTTON_HEIGHT_PX}px !important"
        in dashboard.dashboard_style.value
    )


@pytest.mark.parametrize("mode", ["manual", "fast", "slow"])
def test_all_dashboard_buttons_and_selection_controls_work(tmp_path, monkeypatch, mode):
    first = tmp_path / "first_mini.mat"
    second = tmp_path / "second_mini.mat"
    _write_recording(first)
    _write_recording(second)
    _install_fake_pipeline(monkeypatch)

    dashboard = EventCurationDashboard(data_path=tmp_path, mode=mode, auto_load=False)
    assert dashboard.recording is None

    # Scan updates the recording selector without running the pipeline.
    dashboard.path_text.value = str(tmp_path)
    dashboard.scan_button.click()
    assert len(dashboard.recording_dropdown.options) == 2

    # Load runs exactly one selected recording and enables every event action.
    dashboard.recording_dropdown.value = str(first)
    dashboard.load_button.click()
    assert dashboard.recording.path == first
    assert all(not button.disabled for button in dashboard.decision_buttons.children)
    assert not dashboard.prev_button.disabled
    assert not dashboard.next_button.disabled
    assert not dashboard.event_slider.disabled
    assert not dashboard.threshold_slider.disabled

    # Yes, No, and Clear all mutate the focused event through their callbacks.
    first_key = dashboard.event_keys[0]
    dashboard.yes_button.click()
    assert dashboard.labels[first_key] == "yes"
    dashboard.no_button.click()
    assert dashboard.labels[first_key] == "no"
    dashboard.clear_button.click()
    assert first_key not in dashboard.labels

    # Navigation buttons and slider keep every panel focused on the same event.
    dashboard.next_button.click()
    assert dashboard.current == 1
    dashboard.prev_button.click()
    assert dashboard.current == 0
    dashboard.event_slider.value = 2
    assert dashboard.current == 2
    assert "Candidate PCA (400 ms)" in dashboard.feature_title.value
    assert dashboard.feature_fig.layout.xaxis.title.text.startswith("PC1")
    assert dashboard.feature_fig.layout.yaxis.title.text.startswith("PC2")

    # Plot clicks use each trace's displayed-to-source event mapping.
    points = type("Points", (), {"point_inds": [1]})()
    dashboard._timeline_clicked(None, points, None)
    assert dashboard.current == int(dashboard.timeline_trace_indices[1])
    dashboard._feature_clicked(None, points, None)
    assert dashboard.current == int(dashboard.visible_indices[1])

    # View buttons filter manual labels and preserve selectable event mappings.
    dashboard._select_event(1)
    dashboard.no_button.click()
    dashboard.view_filter.value = "no"
    assert dashboard.visible_indices.tolist() == [1]
    dashboard.view_filter.value = "all"
    assert dashboard.visible_indices.tolist() == [0, 1, 2]


def test_spatial_trace_bypasses_denoiser_and_uses_pf_curation_dir(tmp_path, monkeypatch):
    pf_dir = _write_spatial_recording(tmp_path / "Data")

    class FailDenoiser:
        def __init__(self, *args, **kwargs):
            raise AssertionError("processed BioHPC traces must not rerun the denoiser")

    monkeypatch.setattr("vnoiser.curation.Denoiser", FailDenoiser)
    dashboard = EventCurationDashboard(
        data_path=pf_dir,
        mode="fast",
        duration_s=None,
        auto_load=False,
    )
    (pf_dir / "detected_events_peaks.pkl").unlink()
    dashboard._load_selected_recording(None)

    assert dashboard.pipeline_cache_status == "loaded processed denoised trace"
    assert dashboard.candidates.indices.tolist() == [800, 2000, 3200]
    assert dashboard.pca_event_indices.tolist() == []
    assert dashboard.curation_dir == pf_dir / ".curation"
    assert dashboard.label_path == pf_dir / ".curation" / "fast_template_curation.json"


def test_spatial_root_populates_hierarchy_one_experiment_at_a_time(tmp_path):
    data_root = tmp_path / "Data"
    pf_dir = _write_spatial_recording(data_root)
    dashboard = EventCurationDashboard(
        data_path=data_root,
        mode="fast",
        duration_s=None,
        auto_load=False,
    )

    assert dashboard.recordings == tuple()
    assert dashboard.hierarchy_controls.layout.display == "flex"
    assert dashboard.load_button.disabled is True
    assert dashboard.animal_dropdown.options[1][0] == "stan1"

    dashboard.animal_dropdown.value = str(data_root / "stan1")
    assert dashboard.experiment_dropdown.options[1][0] == "stan1_expt1"
    assert dashboard.recordings == tuple()

    dashboard.experiment_dropdown.value = str(pf_dir.parent)
    assert len(dashboard.recordings) == 1
    assert dashboard.recording_dropdown.options[0][0].endswith("scan 10 / soma")
    assert dashboard.load_button.disabled is False

    dashboard.load_button.click()
    assert dashboard.recording.metadata["experiment"] == "stan1_expt1"


def test_three_modes_save_independent_curation_files(tmp_path, monkeypatch):
    pf_dir = _write_spatial_recording(tmp_path / "Data")
    monkeypatch.setattr(
        "vnoiser.curation.lowpass_trace",
        lambda trace, fs_hz, cutoff_hz: np.asarray(trace, dtype=float).copy(),
    )
    expected = {
        "manual": "manual_template_curation.json",
        "fast": "fast_template_curation.json",
        "slow": "slow_template_curation.json",
    }
    dashboards = {}
    for mode, filename in expected.items():
        dashboard = EventCurationDashboard(
            data_path=pf_dir,
            mode=mode,
            duration_s=None,
            auto_load=False,
        )
        dashboard._load_selected_recording(None)
        dashboard._set_label("yes" if mode != "slow" else "no")
        dashboards[mode] = dashboard
        assert dashboard.label_path == pf_dir / ".curation" / filename
        assert dashboard.label_path.exists()

    assert dashboards["manual"].labels != dashboards["slow"].labels
    assert len(list((pf_dir / ".curation").glob("*_template_curation.json"))) == 3


def test_slow_mode_uses_less_than_40_hz_trace_and_wide_template(tmp_path, monkeypatch):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)
    _install_fake_pipeline(monkeypatch, event_indices=(700, 1500, 2300))

    dashboard = EventCurationDashboard(
        data_path=recording,
        mode="slow",
        auto_load=False,
    )
    dashboard._load_selected_recording(None)

    assert dashboard.candidates.template_time_ms[0] == pytest.approx(-500.0)
    assert dashboard.candidates.template_time_ms[-1] == pytest.approx(500.0)
    assert dashboard.candidate_fig.layout.xaxis.range == (-500, 500)
    assert [trace.name for trace in dashboard.timeline_fig.data[:3]] == [
        "denoised trace",
        "<40 Hz low-pass trace",
        "candidate threshold",
    ]
    assert dashboard.timeline_fig.layout.legend.yanchor == "top"
    assert dashboard.timeline_fig.layout.legend.y == pytest.approx(0.99)
    assert dashboard.timeline_fig.layout.showlegend is True
    assert dashboard.timeline_fig.data[0].showlegend is True
    assert dashboard.timeline_fig.data[1].showlegend is True

    fs_hz = 1000.0
    t = np.arange(4000) / fs_hz
    low = np.sin(2 * np.pi * 5 * t)
    mixed = low + np.sin(2 * np.pi * 120 * t)
    filtered = lowpass_trace(mixed, fs_hz, cutoff_hz=40.0)
    assert np.corrcoef(filtered[100:-100], low[100:-100])[0, 1] > 0.99


def test_threshold_event_calls_uses_selected_curve_and_threshold():
    fs_hz = 1000.0
    trace = np.zeros(1000)
    trace[[100, 105, 500]] = [3.0, 5.0, 4.0]

    peaks = threshold_event_calls(
        trace,
        fs_hz,
        threshold=3.5,
        min_distance_ms=10.0,
    )

    assert peaks.tolist() == [105, 500]


def test_threshold_slider_scale_is_data_derived_and_robust_to_one_outlier():
    fs_hz = 1000.0
    t = np.arange(10000, dtype=float) / fs_hz
    trace = 0.2 * np.sin(2 * np.pi * 12 * t)
    trace[5000] = 100.0

    lower, upper, selected, step = threshold_slider_scale(
        trace,
        fs_hz,
        min_distance_ms=6.0,
    )

    assert lower == pytest.approx(0.0, abs=1e-3)
    assert 0.0 < selected < upper
    assert upper < 5.0
    assert step == pytest.approx((upper - lower) / 100.0)


def test_candidate_pca_embedding_uses_centered_400_ms_only():
    time_ms = np.linspace(-500.0, 500.0, 1001)
    snippets = np.zeros((3, len(time_ms)))
    snippets[0, time_ms < -250.0] = 10.0
    snippets[1, time_ms > 250.0] = -12.0

    scores, explained = candidate_pca_embedding(snippets, time_ms)
    assert np.allclose(scores, 0.0)
    assert np.allclose(explained, 0.0)

    inside = np.abs(time_ms) <= 200.0
    snippets[0, inside] = np.sin(time_ms[inside] / 30.0)
    snippets[1, inside] = np.cos(time_ms[inside] / 40.0)
    scores, explained = candidate_pca_embedding(snippets, time_ms)
    assert scores.shape == (3, 2)
    assert not np.allclose(scores, 0.0)
    assert np.all(np.isfinite(scores))
    assert explained.sum() == pytest.approx(1.0)


def test_fast_threshold_rebuild_retains_manual_label_until_clear(tmp_path):
    pf_dir = _write_spatial_recording(tmp_path / "Data")
    dashboard = EventCurationDashboard(
        data_path=pf_dir,
        mode="fast",
        duration_s=None,
        auto_load=False,
    )
    dashboard._load_selected_recording(None)

    assert dashboard.threshold_event_indices.tolist() == [800, 2000, 3200]
    retained_key = dashboard.event_keys[0]
    dashboard._set_label("yes")

    dashboard.threshold_slider.value = 6.0

    assert dashboard.threshold_event_indices.tolist() == [2000]
    assert dashboard.candidates.indices.tolist() == [800, 2000]
    assert retained_key in dashboard.event_keys
    assert dashboard.labels[retained_key] == "yes"
    assert dashboard.retained_event_indices.tolist() == [800]

    payload = json.loads(dashboard.label_path.read_text())
    assert payload["candidate_detection"]["source"] == "denoised_trace"
    recording_id = dashboard._recording_label_id(dashboard.recording.path)
    assert payload["candidate_detection"]["thresholds"][recording_id] == pytest.approx(
        6.0
    )
    assert payload["events"][retained_key]["candidate_source"] == "retained_manual"

    reloaded = EventCurationDashboard(
        data_path=pf_dir,
        mode="fast",
        duration_s=None,
        auto_load=False,
    )
    reloaded._load_selected_recording(None)
    assert reloaded.candidate_threshold == pytest.approx(6.0)
    assert reloaded.candidates.indices.tolist() == [800, 2000]
    assert reloaded.labels[retained_key] == "yes"

    reloaded._select_event(reloaded.event_keys.index(retained_key))
    reloaded._set_label("unlabeled")
    assert reloaded.candidates.indices.tolist() == [2000]
    assert retained_key not in reloaded.labels
    assert retained_key not in json.loads(reloaded.label_path.read_text())["events"]


def test_thresholds_are_saved_and_restored_per_recording(tmp_path, monkeypatch):
    first = tmp_path / "first_mini.mat"
    second = tmp_path / "second_mini.mat"
    _write_recording(first)
    _write_recording(second)
    _install_fake_pipeline(monkeypatch)

    dashboard = EventCurationDashboard(
        data_path=tmp_path,
        mode="fast",
        auto_load=False,
    )
    dashboard.recording_dropdown.value = str(first)
    dashboard._load_selected_recording(None)
    dashboard.threshold_slider.value = 3.0
    first_threshold = dashboard.candidate_threshold
    first_id = dashboard._recording_label_id(dashboard.recording.path)

    dashboard.recording_dropdown.value = str(second)
    dashboard._load_selected_recording(None)
    assert dashboard.candidate_threshold != pytest.approx(first_threshold)
    dashboard.threshold_slider.value = 4.5
    second_threshold = dashboard.candidate_threshold
    second_id = dashboard._recording_label_id(dashboard.recording.path)

    payload = json.loads(dashboard.label_path.read_text())
    assert payload["candidate_detection"]["thresholds"] == pytest.approx(
        {first_id: first_threshold, second_id: second_threshold}
    )

    dashboard.recording_dropdown.value = str(first)
    dashboard._load_selected_recording(None)
    assert dashboard.candidate_threshold == pytest.approx(first_threshold)


def test_threshold_can_recover_after_zero_candidates(tmp_path):
    pf_dir = _write_spatial_recording(tmp_path / "Data")
    dashboard = EventCurationDashboard(
        data_path=pf_dir,
        mode="fast",
        duration_s=None,
        auto_load=False,
    )
    dashboard._load_selected_recording(None)

    dashboard.threshold_slider.value = dashboard.threshold_slider.max
    assert len(dashboard.candidates.indices) == 0
    assert dashboard.threshold_slider.disabled is False

    dashboard.threshold_slider.value = 2.0
    assert dashboard.candidates.indices.tolist() == [800, 2000, 3200]


def test_seed_template_uses_top_quarter_of_threshold_candidates(tmp_path, monkeypatch):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)
    event_indices = tuple(range(600, 2200, 200))
    _install_fake_pipeline(monkeypatch, event_indices=event_indices)

    dashboard = EventCurationDashboard(
        data_path=recording,
        mode="fast",
        auto_load=False,
    )
    dashboard._load_selected_recording(None)

    assert len(dashboard.threshold_event_indices) == 8
    assert len(dashboard.seed_indices) == 2
    assert dashboard.candidates.indices[dashboard.seed_indices].tolist() == [1800, 2000]


def test_slow_mode_thresholds_direct_lowpass_trace(tmp_path):
    fs_hz = 1000.0
    t = np.arange(4000, dtype=float) / fs_hz
    trace = 0.02 * np.sin(2 * np.pi * 5 * t)
    trace[800] += 10.0
    trace += 4.0 * np.exp(-0.5 * ((t - 2.0) / 0.050) ** 2)
    pf_dir = _write_spatial_recording(
        tmp_path / "Data",
        trace=trace,
        catalog_indices=(800,),
    )

    fast = EventCurationDashboard(
        data_path=pf_dir,
        mode="fast",
        duration_s=None,
        fast_threshold=2.0,
        auto_load=False,
    )
    fast._load_selected_recording(None)
    slow = EventCurationDashboard(
        data_path=pf_dir,
        mode="slow",
        duration_s=None,
        slow_threshold=2.0,
        auto_load=False,
    )
    slow._load_selected_recording(None)

    assert np.any(np.abs(fast.threshold_event_indices - 800) <= 1)
    assert np.any(np.abs(fast.threshold_event_indices - 2000) <= 2)
    assert not np.any(np.abs(slow.threshold_event_indices - 800) <= 25)
    assert np.any(np.abs(slow.threshold_event_indices - 2000) <= 5)
    assert not np.allclose(slow.analysis_trace, slow.denoised)
    assert [trace.name for trace in fast.timeline_fig.data[:2]] == [
        "denoised trace",
        "candidate threshold",
    ]
    assert [trace.name for trace in slow.timeline_fig.data[:3]] == [
        "denoised trace",
        "<40 Hz low-pass trace",
        "candidate threshold",
    ]


def test_consolidated_notebook_contains_all_three_sections():
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo_root / "notebooks" / "curation.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert 'mode="manual"' in source
    assert 'mode="fast"' in source
    assert 'mode="slow"' in source
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("mode", ["fast", "slow"])
def test_auto_pass_amplitude_overrides_waveform_rejection(tmp_path, monkeypatch, mode):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)
    _install_fake_pipeline(monkeypatch, event_indices=(700, 1500, 2300))

    dashboard = EventCurationDashboard(data_path=tmp_path, mode=mode, auto_load=False)
    dashboard._load_selected_recording(None)
    assert not dashboard.auto_pass_slider.disabled
    assert not dashboard.waveform_rejection_checkbox.disabled
    assert dashboard.auto_pass_amplitude is None

    # Force every candidate below the cosine threshold so they auto-reject.
    dashboard.initial_template_scores = np.zeros(len(dashboard.event_keys))
    assert [dashboard._initial_auto_call_for_index(i) for i in range(3)] == [
        "reject",
        "reject",
        "reject",
    ]

    # Amplitudes are 4, 5, 6; auto-pass at 5 passes the top two only.
    dashboard.auto_pass_slider.value = 5.0
    dashboard.initial_template_scores = np.zeros(len(dashboard.event_keys))
    assert dashboard.auto_pass_amplitude == pytest.approx(5.0)
    assert [dashboard._initial_auto_call_for_index(i) for i in range(3)] == [
        "reject",
        "pass",
        "pass",
    ]
    assert dashboard._label_for_index(0) == "auto_no"
    assert dashboard._label_for_index(2) == "auto_yes"
    assert "auto-pass amplitude" in [trace.name for trace in dashboard.timeline_fig.data]

    # Manual labels still win over the amplitude rule.
    dashboard._select_event(2)
    dashboard.no_button.click()
    assert dashboard._label_for_index(2) == "no"


@pytest.mark.parametrize("mode", ["fast", "slow"])
def test_waveform_rejection_toggle_disables_auto_reject(tmp_path, monkeypatch, mode):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)
    _install_fake_pipeline(monkeypatch, event_indices=(700, 1500, 2300))

    dashboard = EventCurationDashboard(data_path=tmp_path, mode=mode, auto_load=False)
    dashboard._load_selected_recording(None)
    dashboard.initial_template_scores = np.array([0.0, 0.0, 1.0])
    assert dashboard._initial_auto_call_for_index(0) == "reject"

    dashboard.waveform_rejection_checkbox.value = False
    dashboard.initial_template_scores = np.array([0.0, 0.0, 1.0])
    assert dashboard.waveform_rejection is False
    assert dashboard._initial_auto_call_for_index(0) is None
    assert dashboard._label_for_index(0) == "unlabeled"
    assert dashboard._initial_auto_call_for_index(2) == "pass"

    dashboard.waveform_rejection_checkbox.value = True
    dashboard.initial_template_scores = np.array([0.0, 0.0, 1.0])
    assert dashboard._initial_auto_call_for_index(0) == "reject"


def test_auto_pass_settings_are_saved_and_restored_per_recording(tmp_path, monkeypatch):
    first = tmp_path / "first_mini.mat"
    second = tmp_path / "second_mini.mat"
    _write_recording(first)
    _write_recording(second)
    _install_fake_pipeline(monkeypatch)

    dashboard = EventCurationDashboard(data_path=tmp_path, mode="fast", auto_load=False)
    dashboard.recording_dropdown.value = str(first)
    dashboard._load_selected_recording(None)
    dashboard.auto_pass_slider.value = 5.0
    dashboard.waveform_rejection_checkbox.value = False
    first_id = dashboard._recording_label_id(dashboard.recording.path)

    dashboard.recording_dropdown.value = str(second)
    dashboard._load_selected_recording(None)
    assert dashboard.auto_pass_amplitude is None
    assert dashboard.waveform_rejection is True
    assert dashboard.auto_pass_slider.value == pytest.approx(dashboard.auto_pass_slider.max)
    second_id = dashboard._recording_label_id(dashboard.recording.path)
    dashboard.auto_pass_slider.value = 4.5

    payload = json.loads(dashboard.label_path.read_text())
    detection = payload["candidate_detection"]
    assert detection["auto_pass_amplitudes"] == pytest.approx({first_id: 5.0, second_id: 4.5})
    assert detection["waveform_rejection"] == {first_id: False, second_id: True}

    reloaded = EventCurationDashboard(data_path=tmp_path, mode="fast", auto_load=False)
    reloaded.recording_dropdown.value = str(first)
    reloaded._load_selected_recording(None)
    assert reloaded.auto_pass_amplitude == pytest.approx(5.0)
    assert reloaded.waveform_rejection is False
    assert reloaded.auto_pass_slider.value == pytest.approx(5.0)
    assert reloaded.waveform_rejection_checkbox.value is False

    reloaded._select_event(0)
    reloaded.yes_button.click()
    event = next(iter(json.loads(reloaded.label_path.read_text())["events"].values()))
    assert event["auto_pass_amplitude"] == pytest.approx(5.0)
    assert event["waveform_rejection"] is False


def test_manual_mode_hides_auto_pass_controls(tmp_path, monkeypatch):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)
    _install_fake_pipeline(monkeypatch)

    dashboard = EventCurationDashboard(data_path=tmp_path, mode="manual", auto_load=False)
    dashboard._load_selected_recording(None)
    assert dashboard.auto_pass_box.layout.display == "none"
    assert dashboard.auto_pass_slider.disabled
    assert dashboard.waveform_rejection_checkbox.disabled
    assert dashboard._initial_auto_call_for_index(0) is None


@pytest.mark.parametrize("mode", ["fast", "slow"])
def test_pc1_line_auto_passes_one_side_of_the_pca(tmp_path, monkeypatch, mode):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)
    _install_fake_pipeline(monkeypatch, event_indices=(700, 1500, 2300))

    dashboard = EventCurationDashboard(data_path=tmp_path, mode=mode, auto_load=False)
    dashboard._load_selected_recording(None)
    assert not dashboard.pc1_slider.disabled
    assert not dashboard.pc1_side.disabled
    assert dashboard.auto_pass_pc1 is None
    # parked at the far end of the passing side: nothing passes yet
    lower, upper, _step = dashboard.pc1_slider_range()
    assert dashboard.pc1_slider.value == pytest.approx(upper)
    pc1 = dashboard.candidates.pca_scores[:, 0]
    assert lower < pc1.min() and upper > pc1.max()

    dashboard.initial_template_scores = np.zeros(3)
    assert [dashboard._initial_auto_call_for_index(i) for i in range(3)] == ["reject"] * 3

    # a line between the two highest PC1 scores passes the one to its right
    order = np.argsort(pc1)
    line = 0.5 * (pc1[order[1]] + pc1[order[2]])
    dashboard.pc1_slider.value = line
    dashboard.initial_template_scores = np.zeros(3)
    assert dashboard.auto_pass_pc1 == pytest.approx(line)
    calls = [dashboard._initial_auto_call_for_index(i) for i in range(3)]
    assert calls[order[2]] == "pass"
    assert calls[order[0]] == calls[order[1]] == "reject"
    assert dashboard._label_for_index(int(order[2])) == "auto_yes"
    assert dashboard.feature_fig.layout.shapes[0].x0 == pytest.approx(line)

    # flipping the side passes the two to its left instead
    dashboard.pc1_side.value = "left"
    dashboard.initial_template_scores = np.zeros(3)
    calls = [dashboard._initial_auto_call_for_index(i) for i in range(3)]
    assert calls[order[2]] == "reject"
    assert calls[order[0]] == calls[order[1]] == "pass"

    # manual labels still win
    dashboard._select_event(int(order[0]))
    dashboard.no_button.click()
    assert dashboard._label_for_index(int(order[0])) == "no"


def test_pc1_range_follows_the_candidates(tmp_path, monkeypatch):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)
    _install_fake_pipeline(monkeypatch, event_indices=(700, 1500, 2300))

    dashboard = EventCurationDashboard(data_path=tmp_path, mode="fast", auto_load=False)
    dashboard._load_selected_recording(None)
    before = (dashboard.pc1_slider.min, dashboard.pc1_slider.max)
    # amplitudes are 4, 5, 6: a threshold of 4.5 drops the first candidate
    # and the PCA is refit on the two left, so the A3 range moves with it
    dashboard.threshold_slider.value = 4.5
    assert len(dashboard.event_keys) == 2
    after = (dashboard.pc1_slider.min, dashboard.pc1_slider.max)
    assert after != before
    pc1 = dashboard.candidates.pca_scores[:, 0]
    assert after[0] < pc1.min() and after[1] > pc1.max()


@pytest.mark.parametrize("mode", ["fast", "slow"])
def test_cosine_slider_sets_the_auto_pass_threshold(tmp_path, monkeypatch, mode):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)
    _install_fake_pipeline(monkeypatch, event_indices=(700, 1500, 2300))

    dashboard = EventCurationDashboard(data_path=tmp_path, mode=mode, auto_load=False)
    dashboard._load_selected_recording(None)
    assert not dashboard.cosine_slider.disabled
    assert dashboard.auto_template_threshold == pytest.approx(0.8)
    assert dashboard.cosine_slider.value == pytest.approx(0.8)

    dashboard.initial_template_scores = np.array([0.5, 0.8, 0.95])
    assert [dashboard._initial_auto_call_for_index(i) for i in range(3)] == [
        "reject",
        "pass",
        "pass",
    ]
    dashboard.cosine_slider.value = 0.9
    dashboard.initial_template_scores = np.array([0.5, 0.8, 0.95])
    assert dashboard.auto_template_threshold == pytest.approx(0.9)
    assert [dashboard._initial_auto_call_for_index(i) for i in range(3)] == [
        "reject",
        "reject",
        "pass",
    ]
    # at the bottom of its range everything passes on cosine alone
    dashboard.cosine_slider.value = -1.0
    dashboard.initial_template_scores = np.array([0.5, 0.8, 0.95])
    assert [dashboard._initial_auto_call_for_index(i) for i in range(3)] == ["pass"] * 3


def test_pc1_and_cosine_settings_are_saved_and_restored_per_recording(tmp_path, monkeypatch):
    first = tmp_path / "first_mini.mat"
    second = tmp_path / "second_mini.mat"
    _write_recording(first)
    _write_recording(second)
    _install_fake_pipeline(monkeypatch)

    dashboard = EventCurationDashboard(data_path=tmp_path, mode="fast", auto_load=False)
    dashboard.recording_dropdown.value = str(first)
    dashboard._load_selected_recording(None)
    lower, upper, _step = dashboard.pc1_slider_range()
    line = 0.5 * (lower + upper)
    dashboard.pc1_slider.value = line
    dashboard.pc1_side.value = "left"
    dashboard.cosine_slider.value = 0.6
    first_id = dashboard._recording_label_id(dashboard.recording.path)

    dashboard.recording_dropdown.value = str(second)
    dashboard._load_selected_recording(None)
    assert dashboard.auto_pass_pc1 is None
    assert dashboard.auto_pass_pc1_side == "right"
    assert dashboard.auto_template_threshold == pytest.approx(0.8)
    second_id = dashboard._recording_label_id(dashboard.recording.path)
    dashboard.cosine_slider.value = 0.7

    detection = json.loads(dashboard.label_path.read_text())["candidate_detection"]
    assert detection["auto_pass_pc1"] == {first_id: pytest.approx(line), second_id: None}
    assert detection["auto_pass_pc1_sides"] == {first_id: "left", second_id: "right"}
    assert detection["auto_template_thresholds"] == pytest.approx({first_id: 0.6, second_id: 0.7})

    reloaded = EventCurationDashboard(data_path=tmp_path, mode="fast", auto_load=False)
    reloaded.recording_dropdown.value = str(first)
    reloaded._load_selected_recording(None)
    assert reloaded.auto_pass_pc1 == pytest.approx(line)
    assert reloaded.auto_pass_pc1_side == "left"
    assert reloaded.auto_template_threshold == pytest.approx(0.6)
    assert reloaded.pc1_slider.value == pytest.approx(line)
    assert reloaded.pc1_side.value == "left"
    assert reloaded.cosine_slider.value == pytest.approx(0.6)

    reloaded._select_event(0)
    reloaded.yes_button.click()
    event = next(iter(json.loads(reloaded.label_path.read_text())["events"].values()))
    assert event["auto_pass_pc1"] == pytest.approx(line)
    assert event["auto_pass_pc1_side"] == "left"
    assert event["auto_template_threshold"] == pytest.approx(0.6)
    assert event["pc1_score"] == pytest.approx(reloaded.candidates.pca_scores[0, 0])


def test_manual_mode_hides_pc1_and_cosine_controls(tmp_path, monkeypatch):
    recording = tmp_path / "sample_mini.mat"
    _write_recording(recording)
    _install_fake_pipeline(monkeypatch)

    dashboard = EventCurationDashboard(data_path=tmp_path, mode="manual", auto_load=False)
    dashboard._load_selected_recording(None)
    assert dashboard.pc1_box.layout.display == "none"
    assert dashboard.cosine_box.layout.display == "none"
    assert dashboard.pc1_slider.disabled
    assert dashboard.cosine_slider.disabled
    assert not dashboard.feature_fig.layout.shapes
