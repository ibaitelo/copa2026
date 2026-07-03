"""
fetch_friendlies_jun2026.py — Amistosos internacionais de seleções abr-jun 2026.

Fontes (API-Football):
  league=10   (International Friendlies — seleções nacionais)
  league=667  (CONCACAF Nations League 2026, se existir na API)
  league=547  (UEFA Nations League Finals 2024/25, se existir)

Período: 2026-04-01 → 2026-06-10 (antes do início da Copa 11/jun)
Apenas times qualificados para a Copa 2026.
match_type="friendly", weight=0.8.

Cache: data/raw/api_cache/ — não faz download se cache existe.
Saída: data/raw/friendlies_jun2026_raw.csv
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(dotenv_path=".env", encoding="utf-8-sig")
API_KEY  = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
RATE_SLEEP = 0.5

CACHE_DIR  = Path("data/raw/api_cache")
OUTPUT_CSV = Path("data/raw/friendlies_jun2026_raw.csv")

DATE_FROM = "2026-04-01"
DATE_TO   = "2026-06-10"   # véspera da Copa (11/jun)

# Ligas a tentar; match_type define como o jogo entra no modelo
COMPETITIONS = [
    {"league": 10,  "season": 2026, "confederation": "FRIENDLY",
     "competition": "International Friendlies 2026", "match_type": "friendly"},
    {"league": 667, "season": 2026, "confederation": "CONCACAF",
     "competition": "CONCACAF Nations League 2026", "match_type": "WCQ"},
    {"league": 547, "season": 2024, "confederation": "UEFA",
     "competition": "UEFA Nations League Finals 2024/25", "match_type": "WCQ"},
]

NAME_MAP = {
    "USA":                  "United States",
    "México":               "Mexico",
    "Curaçao":              "Curacao",
    "Côte d'Ivoire":        "Ivory Coast",
    "Korea Republic":       "South Korea",
    "IR Iran":              "Iran",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Türkiye":              "Turkey",
    "Cape Verde Islands":   "Cape Verde",
    "Congo DR":             "DR Congo",
}

COPA2026_TEAMS = {
    "Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay", "Paraguay",
    "Germany", "France", "Spain", "England", "Netherlands", "Portugal",
    "Belgium", "Austria", "Switzerland", "Croatia",
    "Norway", "Sweden", "Czech Republic", "Turkey",
    "Scotland", "Bosnia and Herzegovina",
    "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia",
    "Uzbekistan", "Jordan", "Iraq", "Qatar",
    "Morocco", "Senegal", "Egypt", "Ghana", "Cape Verde",
    "DR Congo", "Ivory Coast", "South Africa", "Algeria", "Tunisia",
    "United States", "Mexico", "Canada", "Panama", "Curacao", "Haiti",
    "New Zealand",
}

STAT_MAP = {
    "Total Shots":      "shots",
    "Shots on Goal":    "shots_on_goal",
    "Blocked Shots":    "blocked_shots",
    "Ball Possession":  "ball_possession",
    "Fouls":            "fouls",
    "Yellow Cards":     "yellow_cards",
    "Red Cards":        "red_cards",
    "Corner Kicks":     "corners",
    "Offsides":         "offsides",
    "Passes accurate":  "passes_accurate",
    "Goalkeeper Saves": "saves",
}

OUTPUT_COLS = [
    "match_id", "date", "competition", "confederation", "match_type", "season",
    "home_away", "team", "opponent",
    "gols_marcados", "gols_sofridos",
    "shots", "shots_on_goal", "blocked_shots", "ball_possession",
    "fouls", "yellow_cards", "red_cards", "corners", "offsides",
    "passes_accurate", "saves", "ast",
    "has_shot_data", "weight",
]


def _cache_path(endpoint: str, cache_id: str) -> Path:
    p = CACHE_DIR / endpoint.strip("/").replace("/", "_") / f"{cache_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def api_get(endpoint: str, params: dict, cache_id: str) -> dict:
    cp = _cache_path(endpoint, cache_id)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            return json.load(f)
    if not API_KEY:
        return {"response": []}
    headers = {"x-apisports-key": API_KEY}
    url = f"{BASE_URL}/{endpoint.strip('/')}"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
    except Exception:
        # Retry once with SSL verification disabled
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(url, headers=headers, params=params,
                            timeout=30, verify=False)
    resp.raise_for_status()
    data = resp.json()
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    time.sleep(RATE_SLEEP)
    return data


def _norm(name: str) -> str:
    return NAME_MAP.get(name, name)


def _parse_value(val):
    if val is None or val in ("null", ""):
        return None
    if isinstance(val, str):
        stripped = val.rstrip("%").strip()
        return float(stripped) if stripped else None
    return val


def fixture_to_rows(fixture: dict, comp: dict) -> list[dict]:
    f     = fixture["fixture"]
    teams = fixture["teams"]
    goals = fixture["goals"]
    fid   = f["id"]
    date  = f["date"][:10]
    rows  = []
    for side, other in (("home", "away"), ("away", "home")):
        rows.append({
            "match_id":        fid,
            "date":            date,
            "competition":     comp["competition"],
            "confederation":   comp["confederation"],
            "match_type":      comp["match_type"],
            "season":          comp["season"],
            "home_away":       side,
            "team":            _norm(teams[side]["name"]),
            "opponent":        _norm(teams[other]["name"]),
            "gols_marcados":   goals[side],
            "gols_sofridos":   goals[other],
            "shots": None, "shots_on_goal": None, "blocked_shots": None,
            "ball_possession": None, "fouls": None, "yellow_cards": None,
            "red_cards": None, "corners": None, "offsides": None,
            "passes_accurate": None, "saves": None, "ast": None,
            "has_shot_data":   False,
            "weight":          0.8,
        })
    return rows


def enrich_with_stats(rows: list[dict], fixture_id: int) -> None:
    data = api_get("fixtures/statistics", {"fixture": fixture_id},
                   f"stats_{fixture_id}")
    stats_list = data.get("response", [])
    if not stats_list:
        return
    stats_by_team = {
        _norm(e["team"]["name"]): {s["type"]: s["value"] for s in e["statistics"]}
        for e in stats_list
    }
    has_shots = any("Total Shots" in v for v in stats_by_team.values())
    for row in rows:
        ts = stats_by_team.get(row["team"], {})
        for api_key, col in STAT_MAP.items():
            row[col] = _parse_value(ts.get(api_key))
        row["has_shot_data"] = has_shots


def main():
    print("=" * 65)
    print("  fetch_friendlies_jun2026.py — Seleções abr-jun 2026")
    print(f"  Período: {DATE_FROM} → {DATE_TO}")
    print("=" * 65)

    if not API_KEY:
        print("ERRO: API_FOOTBALL_KEY não encontrada em .env")
        sys.exit(1)

    already_done: set = set()
    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        done_df = pd.read_csv(OUTPUT_CSV, usecols=["match_id"])
        already_done = set(done_df["match_id"].astype(str))
        print(f"  Já processados: {len(already_done)} match_ids em cache\n")

    all_rows: list[dict] = []
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    for comp in COMPETITIONS:
        league = comp["league"]
        season = comp["season"]
        cid    = f"league{league}_s{season}_apr_jun2026"
        print(f"\n[League {league} — {comp['competition']}]")

        data     = api_get("fixtures",
                           {"league": league, "season": season,
                            "status": "FT", "from": DATE_FROM, "to": DATE_TO},
                           cid)
        fixtures = data.get("response", [])
        print(f"  {len(fixtures)} jogos FT encontrados")

        copa_fxs = [
            fx for fx in fixtures
            if _norm(fx["teams"]["home"]["name"]) in COPA2026_TEAMS
            or _norm(fx["teams"]["away"]["name"]) in COPA2026_TEAMS
        ]
        print(f"  {len(copa_fxs)} com time Copa 2026")

        new_fxs = [fx for fx in copa_fxs
                   if str(fx["fixture"]["id"]) not in already_done]
        print(f"  {len(new_fxs)} novos para processar\n")

        for fx in new_fxs:
            fid  = fx["fixture"]["id"]
            home = _norm(fx["teams"]["home"]["name"])
            away = _norm(fx["teams"]["away"]["name"])
            date = fx["fixture"]["date"][:10]
            gh   = fx["goals"]["home"]
            ga   = fx["goals"]["away"]
            print(f"  {date}  {home} {gh}–{ga} {away}  (id={fid})")

            rows      = fixture_to_rows(fx, comp)
            enrich_with_stats(rows, fid)
            copa_rows = [r for r in rows if r["team"] in COPA2026_TEAMS]
            all_rows.extend(copa_rows)
            already_done.add(str(fid))

    if not all_rows:
        print("\nNenhum jogo novo de seleção encontrado nestas ligas.")
        print("Verifique os league IDs no API-Football para apr-jun 2026.")
    else:
        df_out = pd.DataFrame(all_rows)
        for col in OUTPUT_COLS:
            if col not in df_out.columns:
                df_out[col] = None
        df_out[OUTPUT_COLS].to_csv(OUTPUT_CSV, mode="a", index=False,
                                   header=write_header, encoding="utf-8")
        print(f"\n  Salvas {len(df_out)} linhas em {OUTPUT_CSV}")

    # Resumo final
    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        df = pd.read_csv(OUTPUT_CSV)
        n_games = df["match_id"].nunique()
        print(f"\n  Resumo: {len(df)} linhas | {n_games} jogos")
        copa_teams_found = sorted(df["team"].unique())
        print(f"  Times Copa: {len(copa_teams_found)}")
        for t in copa_teams_found:
            sub = df[df["team"] == t]
            print(f"    {t}: {len(sub)} linha(s)")
    else:
        print("\n  friendlies_jun2026_raw.csv vazio — nenhum jogo coletado.")


if __name__ == "__main__":
    main()
