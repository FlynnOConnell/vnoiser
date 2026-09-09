import pickle

import numpy as np
from scipy.io import savemat

from vnoiser import JediSub3Dataset, SpatialJediDataset, open_recording_dataset


def test_dataset_samples_window_from_mat_file(tmp_path):
    root = tmp_path / "DS01-JEDI-sub3"
    root.mkdir()
    fs_hz = 1000.0
    t = np.arange(5000, dtype=float) / fs_hz
    trace = np.sin(2 * np.pi * 3 * t)
    events = np.array([250, 1250, 2250, 4250], dtype=int)
    savemat(
        root / "sample_mini.mat",
        {"CAttached": {"fluo_mean": trace, "fluo_time": t, "events_AP": events}},
    )

    dataset = JediSub3Dataset(root)
    sample = dataset.sample(duration_s=2.0, seed=1, require_events=True)

    assert len(dataset) == 1
    assert len(sample.trace) == 2000
    assert sample.t[0] == 0
    assert abs(sample.fs_hz - fs_hz) < 1e-9
    assert ((sample.events_ap_indices >= 0) & (sample.events_ap_indices < 2000)).all()
    assert sample.metadata["filename"] == "sample_mini.mat"


def _write_spatial_dataset(root):
    pf_dir = root / "stan1" / "stan1_expt2_sameCell" / "PF"
    pf_dir.mkdir(parents=True)
    traces = {
        "10": {
            "soma": np.linspace(-1.0, 1.0, 1000),
            "apical1": np.sin(np.linspace(0, 8 * np.pi, 1000)),
        }
    }
    files = {
        "denoised_trace_scans.pkl": traces,
        "fs_scans.pkl": {10: 1000.0},
        "scanIDs_ROIs.pkl": {
            "scanID_spatial": np.array([10]),
            "domain_ROInumber": {"soma": [0], "apical1": [1]},
        },
        "detected_events_peaks.pkl": {
            "10": {"soma": np.array([100, 500, 999]), "apical1": np.array([250])}
        },
    }
    for filename, payload in files.items():
        with (pf_dir / filename).open("wb") as handle:
            pickle.dump(payload, handle)
    return pf_dir


def test_spatial_dataset_indexes_metadata_and_lazily_loads_one_trace(tmp_path):
    data_root = tmp_path / "Data"
    pf_dir = _write_spatial_dataset(data_root)

    dataset = SpatialJediDataset(data_root)

    assert len(dataset) == 0
    assert dataset.animal_options() == [("stan1", str(data_root / "stan1"))]
    experiments = dataset.experiment_options(data_root / "stan1")
    assert experiments == [
        ("stan1_expt2_sameCell", str(pf_dir.parent)),
    ]

    dataset.select_experiment(experiments[0][1])

    assert len(dataset) == 2
    assert dataset.recording_options()[0] == (
        "stan1 / stan1_expt2_sameCell / scan 10 / soma",
        "stan1/stan1_expt2_sameCell/scan=10/domain=soma",
    )

    sample = dataset.load("stan1/stan1_expt2_sameCell/scan=10/domain=soma")

    assert sample.fs_hz == 1000.0
    assert len(sample.trace) == 1000
    assert sample.events_ap_indices.tolist() == [100, 500, 999]
    assert sample.metadata["pre_denoised"] is True
    assert sample.metadata["curation_dir"] == str(pf_dir / ".curation")
    assert sample.metadata["recording_id"].endswith("scan=10/domain=soma")


def test_open_recording_dataset_selects_spatial_format(tmp_path):
    data_root = tmp_path / "Data"
    _write_spatial_dataset(data_root)

    dataset = open_recording_dataset(data_root)

    assert isinstance(dataset, SpatialJediDataset)
    assert dataset.recordings == tuple()


def test_spatial_root_inventory_does_not_open_metadata(tmp_path, monkeypatch):
    data_root = tmp_path / "Data"
    _write_spatial_dataset(data_root)
    SpatialJediDataset.clear_inventory_cache()

    def fail_if_metadata_opens(*args, **kwargs):
        raise AssertionError("root inventory must not open experiment metadata")

    monkeypatch.setattr("vnoiser.dataset._load_small_pickle", fail_if_metadata_opens)
    dataset = SpatialJediDataset(data_root)

    assert dataset.animal_options()[0][0] == "stan1"
    assert dataset.experiment_options(data_root / "stan1")[0][0].startswith("stan1_expt")
    assert dataset.recordings == tuple()
