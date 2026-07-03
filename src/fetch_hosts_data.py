"""
fetch_hosts_data.py — Coleta dados competitivos para os 3 países-sede (USA, México, Canadá).

Esses times foram isentos do WCQ e não estão em api_football_wcq_raw.csv.
Fontes coletadas:
  Copa América 2024:            league=9,   season=2024
  CONCACAF Nations League:      league=536, seasons=2022/2023/2024
  CONCACAF Gold Cup:            league=22,  seasons=2023/2025

Saída: data/raw/hosts_competitive_raw.csv (mesmo schema de api_football_wcq_raw.csv)
Cache: reutiliza data/raw/api_cache/ (retomável)
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
API_KEY = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
RATE_SLEEP = 0.5

CACHE_DIR = Path("data/raw/api_cache")
OUTPUT_CSV = Path("data/raw/hosts_competitive_raw.csv")

# API-Football retorna "USA" para os Estados Unidos (diferente do WCQ que usa "United States")
HOST_TEAMS = {"USA", "United States", "Mexico", "Canada"}
# Normaliza para o nome padrão do projeto
TEAM_NAME_NORM = {"USA": "United States"}

COMPETITIONS = [
    {"name": "Copa America 2024",         "confederation": "CONCACAF", "league": 9,   "season": 2024, "match_type": "tournament"},
    {"name": "CONCACAF Nations League",   "confederation": "CONCACAF", "league": 536, "season": 2024, "match_type": "tournament"},
    {"name": "CONCACAF Nations League",   "confederation": "CONCACAF", "league": 536, "season": 2023, "match_type": "tournament"},
    {"name": "CONCACAF Nations League",   "confederation": "CONCACAF", "league": 536, "season": 2022, "match_type": "tournament"},
    {"name": "CONCACAF Gold Cup",         "confederation": "CONCACAF", "league": 22,  "season": 2025, "match_type": "tournament"},
    {"name": "CONCACAF Gold Cup",         "confederation": "CONCACAF", "league": 22,  "season": 2023, "match_type": "tournament"},
]

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
    "fouls", "yellow_cards", "red_cards", "corners", "offsides", "passes_accurate", "saves",
    "has_shot_data",
]


def _cache_path(endpoint: str, cache_id: str) -> Path:
    p = CACHE_DIR / endpoint.strip("/") / f"{cache_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def api_get(endpoint: str, params: dict, cache_id: str) -> dict:
    cp = _cache_path(endpoint, cache_id)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            return json.load(f)
    headers = {"x-apisports-key": API_KEY}
    resp = requests.get(f"{BASE_URL}/{endpoint.strip('/')}", headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    time.sleep(RATE_SLEEP)
    return data


def _parse_value(val):
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
    f = fixture["fixture"]
    teams = fixture["teams"]
    goals = fixture["goals"]

    fixture_id = f["id"]
    date = f["date"][:10]

    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        row = {
            "match_id":      fixture_id,
            "date":          date,
            "competition":   comp["name"],
            "confederation": comp["confederation"],
            "match_type":    comp["match_type"],
            "season":        comp["season"],
            "home_away":     side,
            "team":          TEAM_NAME_NORM.get(teams[side]["name"], teams[side]["name"]),
            "opponent":      TEAM_NAME_NORM.get(teams[other]["name"], teams[other]["name"]),
            "gols_marcados": goals[side],
            "gols_sofridos": goals[other],
            "shots":         None,
            "shots_on_goal": None,
            "blocked_shots": None,
            "ball_possession": None,
            "fouls":         None,
            "yellow_cards":  None,
            "red_cards":     None,
            "corners":       None,
            "offsides":      None,
            "passes_accurate": None,
            "saves":         None,
            "has_shot_data": False,
        }
        rows.append(row)
    return rows


def enrich_with_stats(rows: list[dict], fixture_id: int) -> None:
    data = api_get("fixtures/statistics", {"fixture": fixture_id}, str(fixture_id))
    stats_list = data.get("response", [])
    if not stats_list:
        return

    # Map team name → stats dict (usando nome normalizado para busca)
    stats_by_team = {}
    for entry in stats_list:
        raw_name = entry["team"]["name"]
        norm_name = TEAM_NAME_NORM.get(raw_name, raw_name)
        stats_by_team[norm_name] = {s["type"]: s["value"] for s in entry["statistics"]}

    has_shots = any("Total Shots" in v for v in stats_by_team.values())

    for row in rows:
        team_stats = stats_by_team.get(row["team"], {})
        for api_key, col in STAT_MAP.items():
            row[col] = _parse_value(team_stats.get(api_key))
        row["has_shot_data"] = has_shots


def main():
    print("=" * 65)
    print("  fetch_hosts_data.py — USA / México / Canadá")
    print("=" * 65 + "\n")

    if not API_KEY:
        print("ERRO: API_FOOTBALL_KEY não encontrada em .env")
        sys.exit(1)

    # Load already-done match_ids
    already_done: set = set()
    if OUTPUT_CSV.exists():
        try:
            done_df = pd.read_csv(OUTPUT_CSV, usecols=["match_id"])
            already_done = set(done_df["match_id"].astype(str).unique())
            print(f"Ja processados: {len(already_done)} match_ids\n")
        except Exception:
            pass

    all_rows: list[dict] = []
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    for comp in COMPETITIONS:
        league = comp["league"]
        season = comp["season"]
        cid = f"league{league}_season{season}_FT"
        print(f"[{comp['name']} - season {season}] Buscando fixtures...")

        data = api_get("fixtures", {"league": league, "season": season, "status": "FT"}, cid)
        fixtures = data.get("response", [])
        print(f"  {len(fixtures)} jogos FT encontrados")

        # Filter: at least one host team involved
        host_fixtures = [
            f for f in fixtures
            if f["teams"]["home"]["name"] in HOST_TEAMS
            or f["teams"]["away"]["name"] in HOST_TEAMS
        ]
        print(f"  {len(host_fixtures)} com time-sede")

        new_fixtures = [f for f in host_fixtures if str(f["fixture"]["id"]) not in already_done]
        print(f"  {len(new_fixtures)} novos (nao em cache de saida)\n")

        for fx in new_fixtures:
            fid = fx["fixture"]["id"]
            home = fx["teams"]["home"]["name"]
            away = fx["teams"]["away"]["name"]
            date = fx["fixture"]["date"][:10]
            print(f"  {date}  {home} vs {away}  (id={fid})")

            rows = fixture_to_rows(fx, comp)
            enrich_with_stats(rows, fid)

            # Keep only rows where team is a host nation
            host_rows = [r for r in rows if r["team"] in HOST_TEAMS]
            all_rows.extend(host_rows)
            already_done.add(str(fid))

    if not all_rows:
        print("\nNenhuma nova linha para salvar.")
    else:
        df_out = pd.DataFrame(all_rows)[OUTPUT_COLS]
        df_out.to_csv(OUTPUT_CSV, mode="a", index=False, header=write_header, encoding="utf-8")
        print(f"\nSalvo {len(df_out)} novas linhas em {OUTPUT_CSV}")

    # Summary
    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        print(f"\n=== Resumo final: {len(df)} linhas totais ===")
        for team in sorted(HOST_TEAMS):
            sub = df[df["team"] == team]
            if len(sub):
                has_stats = sub["shots"].notna().sum()
                print(f"  {team:20s}: {len(sub):3d} jogos, {has_stats:3d} com stats de chutes")
                for comp_name, grp in sub.groupby("competition"):
                    print(f"    {comp_name}: {len(grp)} jogos")


if __name__ == "__main__":
    main()
