"""Single source of truth for seasons, paths, and fetch settings."""
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "data"
RAW_DIR = DATA_DIR / "raw"
EXPORT_DIR = DATA_DIR / "export"
TMP_DIR = WORKSPACE / ".tmp"
DB_PATH = DATA_DIR / "amodel.db"

# --- Multi-league registry --------------------------------------------------
# Single source of truth for supported leagues.
#
#   key              registry key (used in paths, DB, UI)
#   name             display name
#   fd_div           football-data.co.uk division code (I1, E0, ...)
#   understat_slug   Understat league slug (Serie_A, EPL, ...)
#   seasons          seasons to fetch
#   in_progress      season currently in progress (excluded from walk-forward)
#   expected_matches matches in a completed season (380 for 20-team leagues,
#                    306 for the 18-team Bundesliga)
#   pandemic_window  (start, end) of the 2019-20 resumption window, used to
#                    assign summer-2020 matches to the 2019-20 season. None if
#                    the league finished before the summer break.

LEAGUES: dict[str, dict] = {
    "premier_league": {
        "name": "Premier League",
        "fd_div": "E0",
        "understat_slug": "EPL",
        "seasons": [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2027)],
        "in_progress": "2026-27",
        "expected_matches": 380,
        "pandemic_window": None,  # EPL finished in June 2020
    },
    "serie_a": {
        "name": "Serie A",
        "fd_div": "I1",
        "understat_slug": "Serie_A",
        "seasons": [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2027)],
        "in_progress": "2026-27",
        "expected_matches": 380,
        "pandemic_window": ("2020-06-01", "2020-08-31"),
    },
    "la_liga": {
        "name": "La Liga",
        "fd_div": "SP1",
        "understat_slug": "La_Liga",
        "seasons": [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2027)],
        "in_progress": "2026-27",
        "expected_matches": 380,
        "pandemic_window": ("2020-06-01", "2020-08-31"),
    },
    "bundesliga": {
        "name": "Bundesliga",
        "fd_div": "D1",
        "understat_slug": "Bundesliga",
        "seasons": [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2027)],
        "in_progress": "2026-27",
        "expected_matches": 306,  # 18 teams until 2024-25
        "pandemic_window": None,  # Bundesliga finished in June 2020
    },
    "ligue_1": {
        "name": "Ligue 1",
        "fd_div": "F1",
        "understat_slug": "Ligue_1",
        "seasons": [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2027)],
        "in_progress": "2026-27",
        "expected_matches": 380,
        "pandemic_window": ("2020-06-01", "2020-08-31"),
    },
}

DEFAULT_LEAGUE = "serie_a"
LEAGUE_ORDER = list(LEAGUES)  # display order in the UI

# Backward-compatible module-level constants (default league)
SEASONS = LEAGUES[DEFAULT_LEAGUE]["seasons"]
IN_PROGRESS_SEASON = LEAGUES[DEFAULT_LEAGUE]["in_progress"]


def league_dir(source: str, league: str) -> Path:
    """Raw data directory for a league: data/raw/{source}/{league}."""
    return RAW_DIR / source / league


def ml_dir(league: str) -> Path:
    """ML artifact directory for a league: data/export/ml/{league}."""
    return EXPORT_DIR / "ml" / league


def ml_results_dir(league: str) -> Path:
    return ml_dir(league) / "results"


def ml_models_dir(league: str) -> Path:
    return ml_dir(league) / "models"


def max_matchdays(league: str) -> int:
    """Matchdays in a full season (teams - 1), derived from expected_matches.

    expected_matches = teams * (teams - 1) (double round-robin), so
    teams = (1 + sqrt(1 + 4*n)) / 2. 380 -> 20 teams -> 38; 306 -> 18 -> 34.
    """
    n = LEAGUES[league]["expected_matches"]
    teams = int(round((1 + (1 + 4 * n) ** 0.5) / 2))
    return teams - 1


# Legacy flat paths (kept for migration; new code should use the helpers above)
FOOTBALL_DATA_DIR = RAW_DIR / "football_data"
UNDERSTAT_DIR = RAW_DIR / "understat"
ML_DIR = EXPORT_DIR / "ml"
ML_RESULTS_DIR = ML_DIR / "results"
ML_MODELS_DIR = ML_DIR / "models"

# football-data.co.uk URL pattern: https://www.football-data.co.uk/mmz4281/{YYMM}/{div}.csv
# NOTE: Serie A is "I1" (Italy tier 1). "SA0" is actually Scottish Premiership.
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"


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
