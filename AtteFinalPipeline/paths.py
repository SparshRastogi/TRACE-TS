import os
from pathlib import Path

PIPELINE_ROOT = Path(
    os.environ.get("TRACE_PIPELINE_ROOT", Path(__file__).resolve().parents[1])
)
UCIHAR_DATA_DIR = Path(
    os.environ.get("UCI_HAR_NUMPY_DIR", PIPELINE_ROOT / "dataset" / "ucihar")
)
