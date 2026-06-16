"""Centralized project paths for data, models, and script outputs."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "trained_models"
RUNS_DIR = PROJECT_ROOT / "runs"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"
GENERATED_DATA_DIR = OUTPUTS_DIR / "generated_data"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Property dataset filenames (English names; legacy Chinese filenames still supported at load time)
BOILING_POINT_CSV = DATA_DIR / "properties" / "boiling_point.csv"
FORMATION_ENTHALPY_CSV = DATA_DIR / "properties" / "ideal_gas_formation_enthalpy.csv"

LEGACY_BOILING_POINT_CSV = DATA_DIR / "data" / "沸点.csv"
LEGACY_FORMATION_ENTHALPY_CSV = DATA_DIR / "data" / "理想气体生成焓.csv"


def resolve_data_path(preferred: Path, *legacy_paths: Path) -> Path:
    """Return the first existing path among preferred and legacy locations."""
    if preferred.exists():
        return preferred
    for legacy in legacy_paths:
        if legacy.exists():
            return legacy
    return preferred


def ensure_output_dirs() -> None:
    """Create standard output directories if they do not exist."""
    for directory in (
        FIGURES_DIR,
        PREDICTIONS_DIR,
        BENCHMARKS_DIR,
        GENERATED_DATA_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
