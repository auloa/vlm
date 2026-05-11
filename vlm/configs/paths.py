import os
from pathlib import Path

PATH_CONFIG_FILE= Path(__file__).resolve()

VLM_ROOT = PATH_CONFIG_FILE.parent.parent.parent
VLM_MODULE_DIR = VLM_ROOT / VLM_ROOT.name

RUNS_DIR = VLM_ROOT / "training_runs"
os.makedirs(RUNS_DIR, exist_ok=True)