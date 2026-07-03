"""
Coleta dados de WCQ 2026 via API-Football (v3.football.api-sports.io).

Competições cobertas:
  CAF:      league=29, season=2023
  AFC:      league=30, season=2026
  CONCACAF: league=31, season=2026
  UEFA:     league=32, season=2024
  CONMEBOL: league=34, season=2026
  OFC:      league=33, season=2026  (sem statistics — has_shot_data=False)

Uso:
  python src/fetch_api_football.py --test   # valida schema com 3 jogos CONMEBOL
  python src/fetch_api_football.py          # coleta completa

Saída: data/raw/api_football_wcq_raw.csv
Cache: data/raw/api_cache/{endpoint}/{id}.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(encoding="utf-8-sig")
API_KEY = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
RATE_SLEEP = 0.5

CACHE_DIR = Path("data/raw/api_cache")
RAW_DIR = Path("data/raw")

COMPETITIONS = [
    {"name": "CAF WCQ 2026",      "confederation": "CAF",      "league": 29, "season": 2023, "has_stats": True},
    {"name": "AFC WCQ 2026",      "confederation": "AFC",      "league": 30, "season": 2026, "has_stats": True},
    {"name": "CONCACAF WCQ 2026", "confederation": "CONCACAF", "league": 31, "season": 2026, "has_stats": True},
    {"name": "UEFA WCQ 2026",     "confederation": "UEFA",     "league": 32, "season": 2024, "has_stats": True},
    {"name": "CONMEBOL WCQ 2026", "confederation": "CONMEBOL", "league": 34, "season": 2026, "has_stats": True},
    {"name": "OFC WCQ 2026",      "confederation": "OFC",      "league": 33, "season": 2026, "has_stats": False},
]

# Mapeamento API-Football stat type → coluna de saída
STAT_MAP = {
    "Total Shots":     "shots",
    "Shots on Goal":   "shots_on_goal",
    "Blocked Shots":   "blocked_shots",
    "Ball Possession": "ball_possession",
    "Fouls":           "fouls",
    "Yellow Cards":    "yellow_cards",
    "Red Cards":       "red_cards",
    "Corner Kicks":    "corners",
    "Offsides":        "offsides",
    "Passes accurate": "passes_accurate",
    "Goalkeeper Saves":"saves",
}

OUTPUT_COLS = [
    "match_id", "date", "competition", "confederation", "match_type", "season",
    "home_away", "team", "opponent",
    "gols_marcados", "gols_sofridos",
    "shots", "shots_on_goal", "blocked_shots", "ball_possession",
    "fouls", "yellow_cards", "red_cards", "corners", "offsides", "passes_accurate", "saves",
    "has_shot_data",
]

# Times Copa 2026 para o resumo final (nomes como retornados pela API-Football)
COPA2026_TEAMS = {
    # CONMEBOL (6)
    "Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay", "Venezuela",
    # UEFA (16)
    "Germany", "France", "Spain", "England", "Netherlands", "Portugal",
    "Belgium", "Italy", "Austria", "Switzerland", "Croatia", "Denmark",
    "Serbia", "Slovakia", "Slovenia", "Albania",
    # AFC (8)
    "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia",
    "Uzbekistan", "Jordan", "Iraq",
    # CAF (9)
    "Morocco", "Senegal", "Egypt", "Nigeria", "Cameroon",
    "Mali", "Ivory Coast", "South Africa", "Algeria",
    # CONCACAF (6)
    "United States", "Mexico", "Canada", "Panama", "Honduras", "Jamaica",
    # OFC (1)
    "New Zealand",
}


# ---------------------------------------------------------------------------
# Cache + HTTP
# ---------------------------------------------------------------------------

def _cache_path(endpoint: str, cache_id: str) -> Path:
    """data/raw/api_cache/fixtures/statistics/123456.json"""
    p = CACHE_DIR / endpoint.strip("/") / f"{cache_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def api_get(endpoint: str, params: dict, cache_id: str) -> dict:
    """GET from API-Football with local JSON cache. Sleeps only on real requests."""
    cp = _cache_path(endpoint, cache_id)

    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            return json.load(f)

    url = f"{BASE_URL}/{endpoint.strip('/')}"
    headers = {"x-apisports-key": API_KEY}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    time.sleep(RATE_SLEEP)
    return data


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_fixtures(league: int, season: int) -> list:
    data = api_get(
        "fixtures",
        {"league": league, "season": season, "status": "FT"},
        cache_id=f"league{league}_season{season}_FT",
    )
    return data.get("response", [])


def fetch_statistics(fixture_id: int) -> dict:
    return api_get(
        "fixtures/statistics",
        {"fixture": fixture_id},
        cache_id=str(fixture_id),
    )


def fetch_lineups(fixture_id: int) -> dict:
    return api_get(
        "fixtures/lineups",
        {"fixture": fixture_id},
        cache_id=str(fixture_id),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_value(val):
    """Convert API values: None/'null'/'' → None, '56%' → 56.0, else numeric if possible."""
    if val is None or val == "null" or val == "":
        return None
    if isinstance(val, str):
        stripped = val.rstrip("%").strip()
        if stripped == "":
            return None
        try:
            return float(stripped)
        except ValueError:
            return val
    return val


def fixture_to_rows(fixture: dict, comp: dict) -> list[dict]:
    """Build two skeleton rows (home + away) from a fixture object."""
    f = fixture["fixture"]
    teams = fixture["teams"]
    goals = fixture["goals"]

    fixture_id = f["id"]
    date = f["date"][:10]

    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        row = {
            "match_id":       fixture_id,
            "date":           date,
            "competition":    comp["name"],
            "confederation":  comp["confederation"],
            "match_type":     comp.get("match_type", "WCQ"),
            "season":         comp["season"],
            "home_away":      side,
            "team":           teams[side]["name"],
            "opponent":       teams[other]["name"],
            "gols_marcados":  goals[side],
            "gols_sofridos":  goals[other],
            "shots":          None,
            "shots_on_goal":  None,
            "blocked_shots":  None,
            "ball_possession": None,
            "fouls":          None,
            "yellow_cards":   None,
            "red_cards":      None,
            "corners":        None,
            "offsides":       None,
            "passes_accurate": None,
            "saves":          None,
            "has_shot_data":  False,
        }
        rows.append(row)
    return rows


def apply_statistics(rows: list[dict], stats_data: dict) -> None:
    """Fill stat columns in-place using /fixtures/statistics response."""
    team_stats: dict[str, dict] = {}
    for entry in stats_data.get("response", []):
        name = entry["team"]["name"]
        team_stats[name] = {s["type"]: s["value"] for s in entry.get("statistics", [])}

    for row in rows:
        stats = team_stats.get(row["team"])
        if not stats:
            continue
        row["has_shot_data"] = True
        for api_type, col in STAT_MAP.items():
            row[col] = _parse_value(stats.get(api_type))


# ---------------------------------------------------------------------------
# Per-competition collection
# ---------------------------------------------------------------------------

def process_competition(comp: dict, test_limit: int | None = None) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  {comp['name']}  (league={comp['league']}, season={comp['season']})")
    print(f"{'='*60}")

    fixtures = fetch_fixtures(comp["league"], comp["season"])
    print(f"  Fixtures FT: {len(fixtures)}")

    if test_limit:
        fixtures = fixtures[:test_limit]
        print(f"  [TESTE] Limitando a {test_limit} jogos")

    all_rows: list[dict] = []
    for i, fixture in enumerate(fixtures):
        fid = fixture["fixture"]["id"]
        home = fixture["teams"]["home"]["name"]
        away = fixture["teams"]["away"]["name"]
        date = fixture["fixture"]["date"][:10]
        cached_stats = _cache_path("fixtures/statistics", str(fid)).exists()
        cached_lineup = _cache_path("fixtures/lineups", str(fid)).exists()
        print(
            f"  [{i+1:>3}/{len(fixtures)}] {date}  {home} vs {away}  (id={fid})"
            + ("  [stats-cached]" if cached_stats else "")
        )

        rows = fixture_to_rows(fixture, comp)

        if comp["has_stats"]:
            stats_data = fetch_statistics(fid)
            apply_statistics(rows, stats_data)

        # Cache lineups even if not in output schema (useful for future features)
        if not cached_lineup:
            fetch_lineups(fid)

        all_rows.extend(rows)

    return all_rows


# ---------------------------------------------------------------------------
# Competições extras: co-hosts sem WCQ (USA, México, Canada)
# ---------------------------------------------------------------------------
EXTRA_COMPETITIONS = [
    {"name": "CONCACAF Nations League 2023/24", "confederation": "CONCACAF",
     "league": 667, "season": 2023, "has_stats": True, "match_type": "tournament"},
    {"name": "CONCACAF Gold Cup 2023",           "confederation": "CONCACAF",
     "league": 650, "season": 2023, "has_stats": True, "match_type": "tournament"},
]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("  RESUMO")
    print(f"{'='*60}")

    print("\n  Total de jogos por confederação:")
    home_df = df[df["home_away"] == "home"]
    for conf, grp in home_df.groupby("confederation"):
        print(f"    {conf:<12}: {len(grp):>4} jogos")
    print(f"    {'TOTAL':<12}: {len(home_df):>4} jogos")

    print("\n  % jogos com shots disponível por confederação:")
    for conf, grp in home_df.groupby("confederation"):
        has = grp["has_shot_data"].sum()
        pct = has / len(grp) * 100 if len(grp) else 0
        print(f"    {conf:<12}: {pct:>5.1f}%  ({has}/{len(grp)})")

    covered = set(df["team"].dropna().unique())
    matched = COPA2026_TEAMS & covered
    print(f"\n  Times únicos no dataset: {len(covered)}")
    print(f"  Times Copa 2026 (approx 46) cobertos: {len(matched)}/46")
    possibly_missing = COPA2026_TEAMS - covered
    if possibly_missing:
        print(f"  Possivelmente ausentes (nomes podem divergir): {sorted(possibly_missing)}")


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

def run_test() -> None:
    comp = next(c for c in COMPETITIONS if c["confederation"] == "CONMEBOL")
    print(f"\n[TESTE] Buscando fixtures CONMEBOL WCQ (league={comp['league']}, season={comp['season']})...")

    fixtures = fetch_fixtures(comp["league"], comp["season"])
    print(f"  Total fixtures FT: {len(fixtures)}")

    if not fixtures:
        print("ERRO: nenhum fixture retornado. Verifique API_FOOTBALL_KEY e IDs.")
        return

    first_fid = fixtures[0]["fixture"]["id"]
    first_home = fixtures[0]["teams"]["home"]["name"]
    first_away = fixtures[0]["teams"]["away"]["name"]
    print(f"\n  Primeiro jogo: {first_home} vs {first_away} (fixture_id={first_fid})")

    print(f"\n{'─'*60}")
    print(f"  JSON bruto /fixtures/statistics?fixture={first_fid}")
    print(f"{'─'*60}")
    stats_raw = fetch_statistics(first_fid)
    print(json.dumps(stats_raw, indent=2, ensure_ascii=False))

    print(f"\n{'─'*60}")
    print("  Primeiros 3 jogos parseados (schema de saída)")
    print(f"{'─'*60}")
    rows = process_competition(comp, test_limit=3)
    df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    print(f"\n{'='*60}")
    print("  Schema OK? Rodar sem --test para coleta completa:")
    print("  python src/fetch_api_football.py")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta WCQ 2026 via API-Football")
    parser.add_argument("--test",   action="store_true",
                        help="Testa com 3 jogos CONMEBOL e imprime JSON bruto")
    parser.add_argument("--extras", action="store_true",
                        help="Coleta CONCACAF Nations League + Gold Cup para co-hosts")
    args = parser.parse_args()

    if not API_KEY:
        print("ERRO: API_FOOTBALL_KEY não encontrado no .env")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.test:
        run_test()
        return

    if args.extras:
        print("=" * 60)
        print("  Coletando CONCACAF extras (co-hosts)")
        print("=" * 60)
        all_rows: list[dict] = []
        for comp in EXTRA_COMPETITIONS:
            all_rows.extend(process_competition(comp))
        df = pd.DataFrame(all_rows, columns=OUTPUT_COLS)
        out_path = RAW_DIR / "concacaf_extra_raw.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\nSalvo: {out_path}  ({len(df):,} linhas, {df['match_id'].nunique()} jogos únicos)")
        print_summary(df)
        return

    # Coleta completa WCQ
    all_rows_wcq: list[dict] = []
    for comp in COMPETITIONS:
        all_rows_wcq.extend(process_competition(comp))

    df = pd.DataFrame(all_rows_wcq, columns=OUTPUT_COLS)
    out_path = RAW_DIR / "api_football_wcq_raw.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSalvo: {out_path}  ({len(df):,} linhas, {df['match_id'].nunique()} jogos únicos)")

    print_summary(df)


if __name__ == "__main__":
    main()
