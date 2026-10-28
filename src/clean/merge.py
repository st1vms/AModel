"""Merge football-data.co.uk CSVs and Understat JSON into a unified match frame.

Priority rules:
- Scores / shots / referee / HT: football-data.co.uk (primary).
- xG: prefer football-data.co.uk when present (only 2026-27 has HxG/AxG);
  fall back to Understat for every other season.
- Alignment key is (season_file, home, away): both sources are organized by
  season file, and the file's season is authoritative (a date heuristic would
  misassign the pandemic 2019-20 season, whose final matches were played in
  Aug 2020). The two sources also occasionally disagree on the match date by a
  day, which a date-based key would break.
- Matches present in only one source are kept and flagged.
"""
import json

import pandas as pd

import config
from src.clean.normalize import (
    canonical_team,
    parse_date,
    to_float_or_none,
    to_int_or_none,
    unknown_teams,
)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM artifacts from column names.

    Older files have no BOM; 2024-25+ files carry a double-encoded BOM
    (bytes c3af c2bb c2bf = 'ï»¿') that survives utf-8-sig decoding and
    prefixes the first column name.
    """
    for col in list(df.columns):
        cleaned = str(col).lstrip("\ufeff").lstrip("ï»¿").strip()
        if cleaned != col:
            df = df.rename(columns={col: cleaned})
    return df


def load_football_data(league: str = config.DEFAULT_LEAGUE) -> pd.DataFrame:
    """Load all cached football-data.co.uk CSVs for a league into one DataFrame."""
    frames = []
    base = config.league_dir("football_data", league)
    for season in config.LEAGUES[league]["seasons"]:
        path = base / f"{season}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = _clean_columns(df)
        df["season_file"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def _understat_match_fields(m: dict) -> dict | None:
    """Normalize one Understat match dict (new `dates` or legacy `matches`)."""
    # New format: h/a objects, goals/xG as {h,a}, datetime string.
    if isinstance(m.get("h"), dict) and isinstance(m.get("a"), dict):
        dt = m.get("datetime")
        date = dt.split(" ")[0] if dt else None
        return {
            "date": date,
            "home": m["h"].get("title"),
            "away": m["a"].get("title"),
            "goals_home": (m.get("goals") or {}).get("h"),
            "goals_away": (m.get("goals") or {}).get("a"),
            "xg_home": (m.get("xG") or {}).get("h"),
            "xg_away": (m.get("xG") or {}).get("a"),
            "shots_home": m.get("shots"),
            "shots_away": m.get("opponentShots"),
        }
    # Legacy format: flat team/opponent fields.
    if m.get("team") is not None:
        return {
            "date": m.get("date"),
            "home": m.get("team"),
            "away": m.get("opponent"),
            "goals_home": m.get("goalsFor"),
            "goals_away": m.get("opponentGoalsFor"),
            "xg_home": m.get("xG"),
            "xg_away": m.get("opponentxG"),
            "shots_home": m.get("shots"),
            "shots_away": m.get("opponentShots"),
        }
    return None


def load_understat(league: str = config.DEFAULT_LEAGUE) -> pd.DataFrame:
    """Load all cached Understat JSONs for a league into one flat DataFrame."""
    rows = []
    base = config.league_dir("understat", league)
    for season in config.LEAGUES[league]["seasons"]:
        year = config.understat_year(season)
        path = base / f"{year}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        matches = data.get("dates") or data.get("matches") or []
        for m in matches:
            # Skip scheduled (not-yet-played) matches. `isResult` is present in
            # the modern `dates` format; the legacy `matches` format has no such
            # flag, so a missing value is kept.
            if m.get("isResult") is False:
                continue
            fields = _understat_match_fields(m)
            if fields is None:
                continue
            fields["season_file"] = season
            rows.append(fields)
    return pd.DataFrame(rows)


def merge(league: str = config.DEFAULT_LEAGUE) -> pd.DataFrame:
    """Merge both sources for a league into a unified match DataFrame.

    Columns: league, season, date, home_team, away_team, home_goals, away_goals,
    ht_home_goals, ht_away_goals, shots_home, shots_away,
    shots_on_target_home, shots_on_target_away, xg_home, xg_away,
    b365h, b365d, b365a, referee, source, xg_source, flag
    """
    fd = load_football_data(league)
    us = load_understat(league)

    records = {}
    unknown = set()

    # --- football-data.co.uk (primary) ---
    if not fd.empty:
        fd = fd.copy()
        fd["date_iso"] = fd["Date"].map(parse_date)
        fd["home"] = fd["HomeTeam"].map(lambda n: canonical_team(n, league))
        fd["away"] = fd["AwayTeam"].map(lambda n: canonical_team(n, league))
        unknown |= unknown_teams(list(fd["HomeTeam"]) + list(fd["AwayTeam"]), league)
        for _, r in fd.iterrows():
            if r["date_iso"] is None:
                continue
            # The source file's season is authoritative. A date heuristic
            # (season_from_date) misassigns the pandemic 2019-20 season, whose
            # final matches were played in Aug 2020, into 2020-21.
            key = (r["season_file"], r["home"], r["away"])
            has_xg = to_float_or_none(r.get("HxG")) is not None
            records[key] = {
                "league": league,
                "season": r["season_file"],
                "date": r["date_iso"],
                "home_team": r["home"],
                "away_team": r["away"],
                "home_goals": to_int_or_none(r.get("FTHG")),
                "away_goals": to_int_or_none(r.get("FTAG")),
                "ht_home_goals": to_int_or_none(r.get("HTHG")),
                "ht_away_goals": to_int_or_none(r.get("HTAG")),
                "shots_home": to_int_or_none(r.get("HS")),
                "shots_away": to_int_or_none(r.get("AS")),
                "shots_on_target_home": to_int_or_none(r.get("HST")),
                "shots_on_target_away": to_int_or_none(r.get("AST")),
                "xg_home": to_float_or_none(r.get("HxG")),
                "xg_away": to_float_or_none(r.get("AxG")),
                "b365h": to_float_or_none(r.get("B365H")),
                "b365d": to_float_or_none(r.get("B365D")),
                "b365a": to_float_or_none(r.get("B365A")),
                "referee": (str(r.get("Referee")).strip() if pd.notna(r.get("Referee")) else None),
                "source": "football-data.co.uk",
                "xg_source": "football-data.co.uk" if has_xg else None,
                "flag": None,
            }

    # --- Understat (xG fallback + cross-check) ---
    if not us.empty:
        unknown |= unknown_teams(list(us["home"]) + list(us["away"]), league)
        for _, r in us.iterrows():
            if r["date"] is None:
                continue
            # Align on (season_file, home, away): both sources are organized by
            # season file, and the two sources occasionally disagree on the
            # match date by a day, which would break a date-based key.
            key = (r["season_file"], canonical_team(r["home"], league), canonical_team(r["away"], league))
            if key in records:
                rec = records[key]
                # xG fallback for seasons where CSV lacks it
                if rec["xg_home"] is None and to_float_or_none(r["xg_home"]) is not None:
                    rec["xg_home"] = to_float_or_none(r["xg_home"])
                    rec["xg_away"] = to_float_or_none(r["xg_away"])
                    rec["xg_source"] = "understat"
                # cross-source xG agreement check
                if rec["xg_source"] == "football-data.co.uk" and to_float_or_none(r["xg_home"]) is not None:
                    diff = abs(rec["xg_home"] - to_float_or_none(r["xg_home"]))
                    if diff > 0.15:
                        rec["flag"] = f"xg_mismatch_{diff:.2f}"
                # score mismatch check
                if rec["home_goals"] is not None and to_int_or_none(r["goals_home"]) is not None:
                    if rec["home_goals"] != to_int_or_none(r["goals_home"]) or rec["away_goals"] != to_int_or_none(r["goals_away"]):
                        rec["flag"] = (rec["flag"] + ";" if rec["flag"] else "") + "score_mismatch"
            else:
                records[key] = {
                    "league": league,
                    "season": r["season_file"],
                    "date": r["date"],
                    "home_team": canonical_team(r["home"], league),
                    "away_team": canonical_team(r["away"], league),
                    "home_goals": to_int_or_none(r["goals_home"]),
                    "away_goals": to_int_or_none(r["goals_away"]),
                    "ht_home_goals": None,
                    "ht_away_goals": None,
                    "shots_home": to_int_or_none(r["shots_home"]),
                    "shots_away": to_int_or_none(r["shots_away"]),
                    "shots_on_target_home": None,
                    "shots_on_target_away": None,
                    "xg_home": to_float_or_none(r["xg_home"]),
                    "xg_away": to_float_or_none(r["xg_away"]),
                    "b365h": None,
                    "b365d": None,
                    "b365a": None,
                    "referee": None,
                    "source": "understat",
                    "xg_source": "understat" if to_float_or_none(r["xg_home"]) is not None else None,
                    "flag": "understat_only",
                }

    df = pd.DataFrame(list(records.values()))
    if not df.empty:
        df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    if unknown:
        print(f"[warn] unmapped team names: {sorted(unknown)}")
    return df


if __name__ == "__main__":
    import sys

    league = config.DEFAULT_LEAGUE
    if "--league" in sys.argv:
        league = sys.argv[sys.argv.index("--league") + 1]
    df = merge(league)
    print(f"merged {len(df)} matches for {league}")
    print(df.groupby("season").size())
