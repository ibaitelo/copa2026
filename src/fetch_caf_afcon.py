"""
fetch_caf_afcon.py — Coleta dados do AFCON 2023 e 2025 para as 10 seleções africanas qualificadas.

Problema: API-Football não retorna statistics para o CAF WCQ → 8/10 seleções africanas
ficaram com ~5% de completude de stats. AFCON (Copa Africana de Nações) tem statistics
completas e serve como substituto de alta qualidade para caracterizar essas seleções.

Competições:
  AFCON 2023: league=6, season=2023  (realizado em Jan-Feb 2024 na Costa do Marfim)
  AFCON 2025: league=6, season=2025  (realizado em Dez 2025 - Jan 2026 em Marrocos)

Saída: data/raw/caf_afcon_raw.csv (mesmo schema de api_football_wcq_raw.csv)
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
OUTPUT_CSV = Path("data/raw/caf_afcon_raw.csv")

# Seleções africanas qualificadas para a Copa 2026
CAF_QUALIFIED = {
    "Algeria", "Cape Verde Islands", "Congo DR", "Egypt", "Ghana",
    "Ivory Coast", "Morocco", "Senegal", "South Africa", "Tunisia",
}

# Normalização para nomes canônicos do projeto
TEAM_NAME_NORM = {
    "Cape Verde Islands": "Cape Verde",
    "Congo DR":           "DR Congo",
    "Ivory Coast":        "Ivory Coast",  # já OK
}

COMPETITIONS = [
    {"name": "AFCON 2023", "confederation": "CAF", "league": 6, "season": 2023, "match_type": "tournament", "weight": 1.0},
    {"name": "AFCON 2025", "confederation": "CAF", "league": 6, "season": 2025, "match_type": "tournament", "weight": 1.0},
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
        return json.load(open(cp, encoding="utf-8"))
    headers = {"x-apisports-key": API_KEY}
    resp = requests.get(f"{BASE_URL}/{endpoint.strip('/')}", headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    json.dump(data, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
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
        raw_name = teams[side]["name"]
        raw_opp  = teams[other]["name"]
        norm_name = TEAM_NAME_NORM.get(raw_name, raw_name)
        norm_opp  = TEAM_NAME_NORM.get(raw_opp, raw_opp)
        row = {
            "match_id":        fixture_id,
            "date":            date,
            "competition":     comp["name"],
            "confederation":   comp["confederation"],
            "match_type":      comp["match_type"],
            "season":          comp["season"],
            "home_away":       side,
            "team":            norm_name,
            "opponent":        norm_opp,
            "gols_marcados":   goals[side],
            "gols_sofridos":   goals[other],
            "shots":           None, "shots_on_goal":   None, "blocked_shots": None,
            "ball_possession": None, "fouls":           None, "yellow_cards":  None,
            "red_cards":       None, "corners":         None, "offsides":      None,
            "passes_accurate": None, "saves":           None,
            "has_shot_data":   False,
            "_raw_team":       raw_name,  # interno, removido antes de salvar
        }
        rows.append(row)
    return rows


def enrich_with_stats(rows: list[dict], fixture_id: int) -> None:
    data = api_get("fixtures/statistics", {"fixture": fixture_id}, str(fixture_id))
    stats_list = data.get("response", [])
    if not stats_list:
        return

    stats_by_raw = {}
    for entry in stats_list:
        raw_name = entry["team"]["name"]
        stats_by_raw[raw_name] = {s["type"]: s["value"] for s in entry["statistics"]}

    has_shots = any("Total Shots" in v for v in stats_by_raw.values())

    for row in rows:
        team_stats = stats_by_raw.get(row["_raw_team"], {})
        for api_key, col in STAT_MAP.items():
            row[col] = _parse_value(team_stats.get(api_key))
        row["has_shot_data"] = has_shots


def main():
    print("=" * 65)
    print("  fetch_caf_afcon.py — AFCON 2023 & 2025")
    print("=" * 65 + "\n")

    if not API_KEY:
        print("ERRO: API_FOOTBALL_KEY não encontrada em .env")
        sys.exit(1)

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
        league, season = comp["league"], comp["season"]
        cid = f"league{league}_season{season}_FT"
        print(f"[{comp['name']}] Buscando fixtures...")

        data = api_get("fixtures", {"league": league, "season": season, "status": "FT"}, cid)
        fixtures = data.get("response", [])
        print(f"  {len(fixtures)} jogos FT")

        # Filtra: ao menos um time CAF qualificado
        caf_fx = [
            f for f in fixtures
            if f["teams"]["home"]["name"] in CAF_QUALIFIED
            or f["teams"]["away"]["name"] in CAF_QUALIFIED
        ]
        new_fx = [f for f in caf_fx if str(f["fixture"]["id"]) not in already_done]
        print(f"  {len(caf_fx)} com times CAF qualificados, {len(new_fx)} novos\n")

        for fx in new_fx:
            fid = fx["fixture"]["id"]
            home = fx["teams"]["home"]["name"]
            away = fx["teams"]["away"]["name"]
            date = fx["fixture"]["date"][:10]
            print(f"  {date}  {home} vs {away}  (id={fid})")

            rows = fixture_to_rows(fx, comp)
            enrich_with_stats(rows, fid)

            # Mantém apenas times CAF qualificados
            caf_rows = [r for r in rows if r["team"] in {TEAM_NAME_NORM.get(t, t) for t in CAF_QUALIFIED}]
            # Remove coluna interna
            for r in caf_rows:
                r.pop("_raw_team", None)
            all_rows.extend(caf_rows)
            already_done.add(str(fid))

    if all_rows:
        df_out = pd.DataFrame(all_rows)[OUTPUT_COLS]
        df_out.to_csv(OUTPUT_CSV, mode="a", index=False, header=write_header, encoding="utf-8")
        print(f"\nSalvo {len(df_out)} novas linhas em {OUTPUT_CSV}")
    else:
        print("\nNenhuma nova linha para salvar.")

    # Resumo
    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        print(f"\n=== Resumo final: {len(df)} linhas totais ===")
        stat_cols = ["shots", "ball_possession", "passes_accurate"]
        for team in sorted(df["team"].unique()):
            sub = df[df["team"] == team]
            pct = sub["shots"].notna().mean() * 100
            comps = sub["competition"].value_counts().to_dict()
            print(f"  {team:25s}: {len(sub):3d} jogos, {pct:5.1f}% shots  | {comps}")


if __name__ == "__main__":
    main()
