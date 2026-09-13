"""Public API for the vnoiser package."""

from .denoiser import (
    AdaptiveThreshold,
    ClusteringConfig,
    Denoiser,
    FrequencyClusterer,
    WaveletReducer,
    cwtReducerConfig,
    fir_lowpass,
    thresConfig,
    trigger_detect,
)
from .dataset import (
    JediSub3Dataset,
    RecordingSample,
    SpatialJediDataset,
    SpatialTraceRef,
    open_recording_dataset,
)
from .events import SpikeDetectConfig, detect_peaks
from .pf import DomainResult, PfFiles, PfWriter, final_trace, read_pf
from .pipeline import (
    ScanResult,
    ScanTraces,
    load_scan_rois,
    load_vi,
    process_domain,
    process_scan,
    run_from_vi,
    run_pipeline,
)
from .preprocess import DfofConfig, DomainTraces, domain_zscore
from .simulation import Jedi3SubConfig, SimulationResult, simulate_jedi3sub_trace

__all__ = [
    "AdaptiveThreshold",
    "ClusteringConfig",
    "Denoiser",
    "FrequencyClusterer",
    "WaveletReducer",
    "cwtReducerConfig",
    "fir_lowpass",
    "thresConfig",
    "trigger_detect",
    "JediSub3Dataset",
    "RecordingSample",
    "SpatialJediDataset",
    "SpatialTraceRef",
    "open_recording_dataset",
    "SpikeDetectConfig",
    "detect_peaks",
    "DomainResult",
    "PfFiles",
    "PfWriter",
    "final_trace",
    "read_pf",
    "ScanResult",
    "ScanTraces",
    "load_scan_rois",
    "load_vi",
    "process_domain",
    "process_scan",
    "run_from_vi",
    "run_pipeline",
    "DfofConfig",
    "DomainTraces",
    "domain_zscore",
    "Jedi3SubConfig",
    "SimulationResult",
    "simulate_jedi3sub_trace",
]
