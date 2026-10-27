"""Single source of truth for seasons, paths, and fetch settings."""
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "data"
RAW_DIR = DATA_DIR / "raw"
FOOTBALL_DATA_DIR = RAW_DIR / "football_data"
UNDERSTAT_DIR = RAW_DIR / "understat"
EXPORT_DIR = DATA_DIR / "export"
TMP_DIR = WORKSPACE / ".tmp"
DB_PATH = DATA_DIR / "amodel.db"

# Seasons: 2016-17 through 2026-27 (2026-27 in progress as of 2026-10-27)
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2027)]

# Season currently in progress (partial, no 380-match check)
IN_PROGRESS_SEASON = "2026-27"

# football-data.co.uk URL pattern: https://www.football-data.co.uk/mmz4281/{YYMM}/I1.csv
# NOTE: Serie A is "I1" (Italy tier 1). "SA0" is actually Scottish Premiership.
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/I1.csv"


def football_data_code(season: str) -> str:
    """'2016-17' -> '1617'."""
    start, end = season.split("-")
    return f"{start[2:]}{end}"


def understat_year(season: str) -> int:
    """'2016-17' -> 2016 (Understat uses the start year)."""
    return int(season.split("-")[0])


# Politeness: random uniform delay between requests (seconds)
REQUEST_DELAY = (2.0, 5.0)
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0  # seconds, multiplied by attempt number
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Source toggles
USE_FOOTBALL_DATA = True
USE_UNDERSTAT = True
