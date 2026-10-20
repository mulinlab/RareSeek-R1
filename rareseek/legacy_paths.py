"""Shared path convention for source-data scripts.

Environment paths are absolute or relative to the repository, never to cwd.
"""
import os
from .io import path

RAW = path(os.getenv("RARESEEK_RAW_DIR", "RareMed-KG/data/raw"))
PROCESSED = path(os.getenv("RARESEEK_PROCESSED_DIR", "RareMed-KG/data/processed"))
INTEGRATION = PROCESSED / "integration"

def integration_dir():
    INTEGRATION.mkdir(parents=True, exist_ok=True)
    return INTEGRATION
