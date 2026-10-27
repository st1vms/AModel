"""Team-name canonicalization, date parsing, and season derivation.

The canonical map is the main data-quality risk: both football-data.co.uk and
Understat use slightly different club names. We map every known variant to a
single canonical name. Unknown names are logged (not silently dropped) so they
can be added to the map.
"""
import re
from datetime import datetime

# Canonical name -> set of known aliases (lowercased for matching).
# Canonical names are the official club names used in the DB.
_CANONICAL: dict[str, list[str]] = {
    "Inter": ["inter", "inter milan", "internazionale", "fc internazionale milano"],
    "AC Milan": ["milan", "ac milan", "a.c. milan", "associazione calcio milan"],
    "Juventus": ["juventus", "juve", "juventus turin", "juventus fc"],
    "Napoli": ["napoli", "ssc napoli", "naples"],
    "Roma": ["roma", "as roma", "associazione calcio roma"],
    "Lazio": ["lazio", "ss lazio", "lazio roma"],
    "Atalanta": ["atalanta", "atalanta bc", "bc atalanta"],
    "Fiorentina": ["fiorentina", "ac fiorentina", "a.c. fiorentina"],
    "Sampdoria": ["sampdoria", "uc sampdoria", "genoa sampdoria"],
    "Torino": ["torino", "torino fc", "fc torino"],
    "Bologna": ["bologna", "bologna fc", "fc bologna"],
    "Genoa": ["genoa", "genoa cfc", "cfc genoa"],
    "Hellas Verona": ["hellas verona", "hellas verona fc", "verona", "ac hellas verona"],
    "Udinese": ["udinese", "udinese calcio", "ac udinese"],
    "Sassuolo": ["sassuolo", "us sassuolo", "sassuolo calcio"],
    "Empoli": ["empoli", "ac empoli", "empoli fc"],
    "Crotone": ["crotone", "fc crotone", "crotone calcio"],
    "Frosinone": ["frosinone", "frosinone calcio", "us frosinone"],
    "SPAL": ["spal", "spal fc", "fc spal", "spal 2013"],
    "Chievo": ["chievo", "chievo verona", "ac chievo verona"],
    "Palermo": ["palermo", "us palermo", "palermo fc"],
    "Cagliari": ["cagliari", "cagliari calcio", "us cagliari"],
    "Bari": ["bari", "bari calcio", "as bari"],
    "Brescia": ["brescia", "brescia calcio", "ac brescia"],
    "Parma": ["parma", "parma calcio", "parma ac", "parma calcio 1913"],
    "Cosenza": ["cosenza", "cosenza calcio", "us cosenza"],
    "Salernitana": ["salernitana", "us salernitana", "salernitana 1919"],
    "Monza": ["monza", "monza bc", "monza calcio"],
    "Venezia": ["venezia", "venezia fc", "venezia calcio"],
    "Como": ["como", "como 1907", "como calcio"],
    "Lecce": ["lecce", "lecce calcio", "us lecce"],
    "Pisa": ["pisa", "pisa calcio", "ac pisa"],
    "Modena": ["modena", "modena fc", "modena calcio"],
    "Cremonese": ["cremonese", "us cremonese", "cremonese calcio"],
    "Perugia": ["perugia", "perugia calcio", "ac perugia"],
    "Pordenone": ["pordenone", "us pordenone", "pordenone calcio"],
    "Sudtirol": ["sudtirol", "usd sudtirol", "fc sudtirol", "alto adige"],
    "Reggiana": ["reggiana", "reggiana 1919", "reggiana calcio"],
    "Spezia": ["spezia", "spezia calcio", "us spezia"],
    "Virtus Entella": ["virtus entella", "entella", "sv virtus entella"],
    "Avellino": ["avellino", "us avellino", "avellino calcio"],
    "Pescara": ["pescara", "delfino pescara", "pescara calcio"],
    "Trapani": ["trapani", "trapani calcio", "as trapani"],
    "Benevento": ["benevento", "benevento calcio", "us benevento"],
    "Ascoli": ["ascoli", "ascoli calcio", "ascoli picchio"],
    "Cittadella": ["cittadella", "cittadella calcio", "us cittadella"],
    "Frosinone": ["frosinone"],
}

# Build reverse lookup: alias (lower) -> canonical
_ALIAS_TO_CANON: dict[str, str] = {}
for canon, aliases in _CANONICAL.items():
    _ALIAS_TO_CANON[canon.lower()] = canon
    for a in aliases:
        _ALIAS_TO_CANON[a.lower()] = canon


def canonical_team(name: str) -> str:
    """Map a raw team name to its canonical form.

    Falls back to the stripped original (title-cased) if unknown, so nothing is
    silently dropped. Callers should log unknowns via `unknown_teams`.
    """
    if name is None:
        return "UNKNOWN"
    key = str(name).strip().lower()
    return _ALIAS_TO_CANON.get(key, str(name).strip())


def unknown_teams(names: list[str]) -> set[str]:
    """Return the set of names that have no canonical mapping."""
    out = set()
    for n in names:
        if n is None:
            continue
        key = str(n).strip().lower()
        if key not in _ALIAS_TO_CANON:
            out.add(str(n).strip())
    return out


def parse_date(raw: str) -> str | None:
    """Parse a date string (DD/MM/YYYY or YYYY-MM-DD) to ISO YYYY-MM-DD."""
    if raw is None:
        return None
    s = str(raw).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def season_from_date(iso_date: str) -> str:
    """Derive a 'YYYY-YY' season label from an ISO date.

    Serie A season runs Aug->May. A match in Aug/Sept of year Y belongs to
    season Y-(Y+1); a match in Jan-May of year Y belongs to (Y-1)-Y.
    """
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    if dt.month >= 7:  # Aug..Dec -> season starting this year
        start = dt.year
    else:  # Jan..Jun -> season that started last year
        start = dt.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def to_int_or_none(v) -> int | None:
    """Coerce a possibly-empty numeric field to int or None."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def to_float_or_none(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None
