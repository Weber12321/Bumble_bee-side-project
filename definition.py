from pathlib import Path, PurePath

ROOT_DIR = PurePath(__file__).parent

SAVE_FOLDER = Path(ROOT_DIR / "save_models")
Path(SAVE_FOLDER).mkdir(exist_ok=True)

LOGS_DIR = Path(ROOT_DIR / "logs")
Path(LOGS_DIR).mkdir(exist_ok=True)