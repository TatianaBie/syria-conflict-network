from __future__ import annotations

import os
from pathlib import Path


def _read_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _discover_project_root() -> Path:
    repo_root = Path(__file__).resolve().parent
    _read_dotenv(repo_root / ".env")

    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return repo_root


PROJECT_ROOT = _discover_project_root()
DATA_DIR = PROJECT_ROOT / "01 - Data"
SOURCE_DIR = DATA_DIR / "01 - Source_Data"
CLEANED_DIR = DATA_DIR / "02 - Source_Data-cleaned"
MERGING_DIR = DATA_DIR / "Merging"


def data_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts).resolve()


def repo_base_dir() -> Path:
    return PROJECT_ROOT
