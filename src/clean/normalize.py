"""Team-name canonicalization, date parsing, and season derivation.

The canonical map is the main data-quality risk: both football-data.co.uk and
Understat use slightly different club names. We map every known variant to a
single canonical name. Unknown names are logged (not silently dropped) so they
can be added to the map.

Maps are keyed by league so each league's clubs are canonicalized independently.
Leagues without a curated map fall back to the stripped original name.
"""
import re
from datetime import datetime

import config

# Per-league canonical maps: league -> {canonical name -> [aliases]}.
# Canonical names are the official club names used in the DB.
CANONICAL_BY_LEAGUE: dict[str, dict[str, list[str]]] = {
    "serie_a": {
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
    },
    "bundesliga": {
    "Arminia Bielefeld": ["arminia bielefeld", "bielefeld"],
    "Augsburg": ["augsburg"],
    "Bayer Leverkusen": ["bayer leverkusen", "leverkusen"],
    "Bayern Munich": ["bayern munich", "bayern"],
    "Bochum": ["bochum"],
    "Borussia Dortmund": ["borussia dortmund", "dortmund"],
    "Borussia M.Gladbach": ["borussia m.gladbach", "borussia m'gladbach", "m'gladbach", "gladbach"],
    "Darmstadt": ["darmstadt", "darmstadt 98"],
    "Eintracht Frankfurt": ["eintracht frankfurt", "ein frankfurt", "frankfurt"],
    "Elversberg": ["elversberg"],
    "FC Cologne": ["fc cologne", "fc koln", "cologne", "koln"],
    "FC Heidenheim": ["fc heidenheim", "heidenheim"],
    "Fortuna Duesseldorf": ["fortuna duesseldorf", "fortuna dusseldorf", "fortuna düsseldorf"],
    "Freiburg": ["freiburg"],
    "Greuther Fuerth": ["greuther fuerth", "greuther furth", "greuther fürth", "fuerth", "furth"],
    "Hamburger SV": ["hamburger sv", "hamburg"],
    "Hannover 96": ["hannover 96", "hannover"],
    "Hertha Berlin": ["hertha berlin", "hertha"],
    "Hoffenheim": ["hoffenheim"],
    "Holstein Kiel": ["holstein kiel", "kiel"],
    "Ingolstadt": ["ingolstadt", "fc ingolstadt"],
    "Mainz 05": ["mainz 05", "mainz"],
    "Nuernberg": ["nuernberg", "nurnberg", "nürnberg"],
    "Paderborn": ["paderborn", "sc paderborn"],
    "RasenBallsport Leipzig": ["rasenballsport leipzig", "rb leipzig", "leipzig"],
    "Schalke 04": ["schalke 04", "schalke"],
    "St. Pauli": ["st. pauli", "st pauli"],
    "Union Berlin": ["union berlin"],
    "VfB Stuttgart": ["vfb stuttgart", "stuttgart"],
    "Werder Bremen": ["werder bremen", "bremen"],
    "Wolfsburg": ["wolfsburg"],
    },
    "premier_league": {
    "Arsenal": ["arsenal"],
    "Aston Villa": ["aston villa"],
    "Bournemouth": ["bournemouth"],
    "Brentford": ["brentford"],
    "Brighton": ["brighton"],
    "Burnley": ["burnley"],
    "Cardiff": ["cardiff"],
    "Chelsea": ["chelsea"],
    "Coventry": ["coventry"],
    "Crystal Palace": ["crystal palace"],
    "Everton": ["everton"],
    "Fulham": ["fulham"],
    "Huddersfield": ["huddersfield"],
    "Hull": ["hull"],
    "Ipswich": ["ipswich"],
    "Leeds": ["leeds"],
    "Leicester": ["leicester"],
    "Liverpool": ["liverpool"],
    "Luton": ["luton"],
    "Manchester City": ["manchester city", "man city"],
    "Manchester United": ["manchester united", "man united"],
    "Middlesbrough": ["middlesbrough"],
    "Newcastle United": ["newcastle united", "newcastle"],
    "Norwich": ["norwich"],
    "Nottingham Forest": ["nottingham forest", "nott'm forest"],
    "Sheffield United": ["sheffield united"],
    "Southampton": ["southampton"],
    "Stoke": ["stoke"],
    "Sunderland": ["sunderland"],
    "Swansea": ["swansea"],
    "Tottenham": ["tottenham"],
    "Watford": ["watford"],
    "West Bromwich Albion": ["west bromwich albion", "west brom"],
    "West Ham": ["west ham"],
    "Wolverhampton Wanderers": ["wolverhampton wanderers", "wolves"],
    },
    "la_liga": {
    "Alaves": ["alaves"],
    "Almeria": ["almeria"],
    "Athletic Club": ["athletic club", "ath bilbao"],
    "Atletico Madrid": ["atletico madrid", "ath madrid"],
    "Barcelona": ["barcelona"],
    "Cadiz": ["cadiz"],
    "Celta Vigo": ["celta vigo", "celta"],
    "Deportivo La Coruna": ["deportivo la coruna", "la coruna"],
    "Eibar": ["eibar"],
    "Elche": ["elche"],
    "Espanyol": ["espanyol", "espanol"],
    "Getafe": ["getafe"],
    "Girona": ["girona"],
    "Granada": ["granada"],
    "Las Palmas": ["las palmas"],
    "Leganes": ["leganes"],
    "Levante": ["levante"],
    "Malaga": ["malaga"],
    "Mallorca": ["mallorca"],
    "Osasuna": ["osasuna"],
    "Racing Santander": ["racing santander", "santander"],
    "Rayo Vallecano": ["rayo vallecano", "vallecano"],
    "Real Betis": ["real betis", "betis"],
    "Real Madrid": ["real madrid"],
    "Real Oviedo": ["real oviedo", "oviedo"],
    "Real Sociedad": ["real sociedad", "sociedad"],
    "Real Valladolid": ["real valladolid", "valladolid"],
    "SD Huesca": ["sd huesca", "huesca"],
    "Sevilla": ["sevilla"],
    "Sporting Gijon": ["sporting gijon", "sp gijon"],
    "Valencia": ["valencia"],
    "Villarreal": ["villarreal"],
    },
    "ligue_1": {
    "Ajaccio": ["ajaccio"],
    "Amiens": ["amiens"],
    "Angers": ["angers"],
    "Auxerre": ["auxerre"],
    "Bordeaux": ["bordeaux"],
    "Brest": ["brest"],
    "Caen": ["caen"],
    "Clermont Foot": ["clermont foot", "clermont"],
    "Dijon": ["dijon"],
    "Guingamp": ["guingamp"],
    "Le Havre": ["le havre"],
    "Le Mans": ["le mans"],
    "Lens": ["lens"],
    "Lille": ["lille"],
    "Lorient": ["lorient"],
    "Lyon": ["lyon"],
    "Marseille": ["marseille"],
    "Metz": ["metz"],
    "Monaco": ["monaco"],
    "Montpellier": ["montpellier"],
    "Nancy": ["nancy"],
    "Nantes": ["nantes"],
    "Nice": ["nice"],
    "Nimes": ["nimes"],
    "Paris FC": ["paris fc"],
    "Paris Saint Germain": ["paris saint germain", "paris sg"],
    "Reims": ["reims"],
    "Rennes": ["rennes"],
    "SC Bastia": ["sc bastia", "bastia"],
    "Saint-Etienne": ["saint-etienne", "st etienne"],
    "Strasbourg": ["strasbourg"],
    "Toulouse": ["toulouse"],
    "Troyes": ["troyes"],
    },
}

# Build per-league reverse lookup: league -> {alias (lower) -> canonical}
_ALIAS_TO_CANON: dict[str, dict[str, str]] = {}
for _league, _map in CANONICAL_BY_LEAGUE.items():
    _rev: dict[str, str] = {}
    for canon, aliases in _map.items():
        _rev[canon.lower()] = canon
        for a in aliases:
            _rev[a.lower()] = canon
    _ALIAS_TO_CANON[_league] = _rev


def canonical_team(name: str, league: str = config.DEFAULT_LEAGUE) -> str:
    """Map a raw team name to its canonical form for a league.

    Falls back to the stripped original if unknown, so nothing is silently
    dropped. Callers should log unknowns via `unknown_teams`.
    """
    if name is None:
        return "UNKNOWN"
    key = str(name).strip().lower()
    return _ALIAS_TO_CANON.get(league, {}).get(key, str(name).strip())


def unknown_teams(names: list[str], league: str = config.DEFAULT_LEAGUE) -> set[str]:
    """Return the set of names that have no canonical mapping for a league."""
    rev = _ALIAS_TO_CANON.get(league, {})
    out = set()
    for n in names:
        if n is None:
            continue
        key = str(n).strip().lower()
        if key not in rev:
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
