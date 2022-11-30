from pathlib import Path


_MODULE_DIR = Path(__file__).parent
_REPO_ROOT = _MODULE_DIR.parent.parent.parent
DEFAULT_DATADIR_PATH = _REPO_ROOT / "data"
DEFAULT_DATADIR_STR = str(DEFAULT_DATADIR_PATH.resolve().absolute())
