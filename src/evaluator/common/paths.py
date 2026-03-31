"""Common path constants for evaluator code."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
EVALUATOR_ROOT = SRC_ROOT / "evaluator"
RETRIEVAL_ASSETS = PROJECT_ROOT / "dataset" / "retrieval"
CLASSIFICATION_ASSETS = PROJECT_ROOT / "dataset" / "classification"
REFER_ASSETS = PROJECT_ROOT / "dataset" / "refer"
DEFAULT_MODEL_ROOT = SRC_ROOT / "fg-clip"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
