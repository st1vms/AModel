"""Completeness and anomaly validation.

Usage: python -m src.validate.checks
Writes a detailed report to .tmp/validation_report.txt and prints a summary.
Hard errors (exit code 1) vs warnings (exit code 0).
"""
import sqlite3
import sys
from datetime import datetime

import config
from src.db.upsert import get_conn

REPORT_PATH = config.TMP_DIR / "validation_report.txt"


def run_checks(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    seasons = [r[0] for r in conn.execute("SELECT season FROM seasons ORDER BY season")]

    for season in seasons:
        n = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE season = ?", (season,)
        ).fetchone()[0]

        # 1. Completed seasons must have 380 matches
        if season != config.IN_PROGRESS_SEASON and n != 380:
            errors.append(f"{season}: expected 380 matches, found {n}")

        # 2. No duplicate (date, home, away)
        dups = conn.execute(
            """
            SELECT date, home_team_id, away_team_id, COUNT(*) c
            FROM matches WHERE season = ?
            GROUP BY date, home_team_id, away_team_id HAVING c > 1
            """,
            (season,),
        ).fetchall()
        if dups:
            errors.append(f"{season}: {len(dups)} duplicate (date, home, away) rows")

        # 3. FT >= HT consistency
        bad_ht = conn.execute(
            """
            SELECT COUNT(*) FROM matches WHERE season = ?
            AND (ht_home_goals IS NOT NULL AND home_goals IS NOT NULL
                 AND (home_goals < ht_home_goals OR away_goals < ht_away_goals))
            """,
            (season,),
        ).fetchone()[0]
        if bad_ht:
            errors.append(f"{season}: {bad_ht} matches with FT < HT goals")

        # 4. xG sanity: in [0, 8]. The upper bound must allow high-scoring
        # matches (e.g. Inter 6-2 Pisa, xG 6.24); 8 is generous but still
        # catches data-entry errors.
        bad_xg = conn.execute(
            """
            SELECT COUNT(*) FROM matches WHERE season = ?
            AND ((xg_home IS NOT NULL AND (xg_home < 0 OR xg_home > 8))
              OR (xg_away IS NOT NULL AND (xg_away < 0 OR xg_away > 8)))
            """,
            (season,),
        ).fetchone()[0]
        if bad_xg:
            errors.append(f"{season}: {bad_xg} matches with xG outside [0, 8]")

        # 5. xG coverage
        with_xg = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE season = ? AND xg_home IS NOT NULL",
            (season,),
        ).fetchone()[0]
        if with_xg < n:
            warnings.append(f"{season}: xG missing for {n - with_xg}/{n} matches")

        # 6. Score completeness
        no_score = conn.execute(
            """
            SELECT COUNT(*) FROM matches WHERE season = ?
            AND (home_goals IS NULL OR away_goals IS NULL)
            """,
            (season,),
        ).fetchone()[0]
        if no_score:
            errors.append(f"{season}: {no_score} matches without a final score")

        # 7. Cross-source xG mismatches (flagged during merge)
        mism = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE season = ? AND flag LIKE 'xg_mismatch%'",
            (season,),
        ).fetchone()[0]
        if mism:
            warnings.append(f"{season}: {mism} cross-source xG mismatches > 0.15")

        # 8. Single-source-only matches
        only = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE season = ? AND flag = 'understat_only'",
            (season,),
        ).fetchone()[0]
        if only:
            warnings.append(f"{season}: {only} matches present only in Understat")

        # 9. Date range sanity (Aug..May)
        bad_dates = conn.execute(
            """
            SELECT COUNT(*) FROM matches WHERE season = ?
            AND CAST(strftime('%m', date) AS INTEGER) NOT BETWEEN 8 AND 15
            AND CAST(strftime('%m', date) AS INTEGER) NOT BETWEEN 1 AND 5
            """,
            (season,),
        ).fetchone()[0]
        if bad_dates:
            warnings.append(f"{season}: {bad_dates} matches with month outside Aug-May")

    # 10. Global: unknown team names (shouldn't happen after canonicalization)
    unknown = conn.execute(
        "SELECT COUNT(*) FROM teams WHERE name LIKE 'UNKNOWN'"
    ).fetchone()[0]
    if unknown:
        errors.append(f"{unknown} teams with UNKNOWN name")

    return errors, warnings


def main() -> int:
    conn = get_conn()
    errors, warnings = run_checks(conn)
    conn.close()

    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Validation report — {datetime.now().isoformat(timespec='seconds')}",
        f"Seasons checked: {len(config.SEASONS)}",
        "",
        f"HARD ERRORS ({len(errors)}):",
        *([f"  - {e}" for e in errors] or ["  (none)"]),
        "",
        f"WARNINGS ({len(warnings)}):",
        *([f"  - {w}" for w in warnings] or ["  (none)"]),
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nreport: {REPORT_PATH}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
