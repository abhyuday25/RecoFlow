from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"
MOVIES_PATH = RAW_DATA_DIR / "movies.csv"

DEFAULT_TOP_K = 10
