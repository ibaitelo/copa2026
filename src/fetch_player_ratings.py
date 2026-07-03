"""
fetch_player_ratings.py — Ratings de jogadores por jogo para todos os fixtures com stats.

Para cada fixture com has_shot_data=True nas fontes WCQ/HOSTS/AFCON, chama:
  GET /fixtures/players?fixture={id}

Extrai por time por jogo (apenas jogadores que entraram em campo):
  rating_medio    — média dos ratings dos jogadores que jogaram
  rating_max      — maior rating (craque do jogo)
  rating_min      — menor rating (elo fraco)
  n_players_rated — quantos jogadores tiveram rating

Saída: data/raw/player_ratings_per_game.csv
Cache: data/raw/api_cache/fixtures_players/{fixture_id}.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(dotenv_path=".env", encoding="utf-8-sig")
API_KEY  = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
RATE_SLEEP = 0.5

CACHE_DIR  = Path("data/raw/api_cache/fixtures_players")
OUTPUT_CSV = Path("data/raw/player_ratings_per_game.csv")

# Mesmo NAME_MAP de integrate_datasets.py
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

OUTPUT_COLS = ["match_id", "team", "rating_medio", "rating_max",
               "rating_min", "n_players_rated"]


def _norm(name: str) -> str:
    return NAME_MAP.get(name, name)


def _cache_path(fixture_id: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{fixture_id}.json"


def api_get_players(fixture_id: int) -> dict:
    cp = _cache_path(fixture_id)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            return json.load(f)
    headers = {"x-apisports-key": API_KEY}
    resp = requests.get(
        f"{BASE_URL}/fixtures/players",
        headers=headers,
        params={"fixture": fixture_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    time.sleep(RATE_SLEEP)
    return data


def extract_team_ratings(data: dict, fixture_id: int) -> list[dict]:
    """
    Extrai rating médio/max/min por time de uma resposta /fixtures/players.
    Usa apenas jogadores que entraram em campo (minutes > 0).
    """
    rows = []
    for team_entry in data.get("response", []):
        team_name = _norm(team_entry["team"]["name"])
        players   = team_entry.get("players", [])

        ratings = []
        for player in players:
            stats = player.get("statistics", [{}])[0]
            games = stats.get("games", {})
            minutes = games.get("minutes") or 0
            rating  = games.get("rating")

            if not rating or not minutes or minutes <= 0:
                continue
            try:
                r = float(rating)
                if r > 0:
                    ratings.append(r)
            except (ValueError, TypeError):
                continue

        if len(ratings) >= 2:
            rows.append({
                "match_id":        fixture_id,
                "team":            team_name,
                "rating_medio":    round(float(np.mean(ratings)), 4),
                "rating_max":      round(float(max(ratings)), 4),
                "rating_min":      round(float(min(ratings)), 4),
                "n_players_rated": len(ratings),
            })

    return rows


def collect_fixture_ids() -> set[int]:
    """Coleta todos os fixture_ids com has_shot_data=True das fontes existentes."""
    sources = [
        Path("data/raw/api_football_wcq_raw.csv"),
        Path("data/raw/hosts_competitive_raw.csv"),
        Path("data/raw/caf_afcon_raw.csv"),
    ]
    ids: set[int] = set()
    for path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["match_id", "has_shot_data"])
        mask = df["has_shot_data"].map(
            lambda x: str(x).lower() in ("true", "1")
        )
        ids.update(df.loc[mask, "match_id"].astype(int).tolist())
    return ids


def main():
    print("=" * 65)
    print("  fetch_player_ratings.py — Ratings de jogadores por jogo")
    print("=" * 65)

    if not API_KEY:
        print("ERRO: API_FOOTBALL_KEY não encontrada em .env")
        sys.exit(1)

    all_fixture_ids = collect_fixture_ids()
    print(f"\n  Total fixtures com has_shot_data: {len(all_fixture_ids)}")

    # Carregar já processados
    done_fids: set[int] = set()
    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        done_df = pd.read_csv(OUTPUT_CSV, usecols=["match_id"])
        done_fids = set(done_df["match_id"].astype(int).tolist())
        print(f"  Já coletados: {len(done_fids)} fixtures")

    # Fixtures que já têm cache mas não estão no CSV (recuperação)
    cached = {int(p.stem) for p in CACHE_DIR.glob("*.json") if p.stem.isdigit()}
    pending_from_cache = (cached - done_fids) & all_fixture_ids

    to_fetch   = all_fixture_ids - done_fids - cached
    total_work = len(pending_from_cache) + len(to_fetch)
    print(f"  Do cache (processar): {len(pending_from_cache)}")
    print(f"  Da API (buscar):      {len(to_fetch)}")
    print(f"  Total de trabalho:    {total_work}\n")

    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    all_rows: list[dict] = []
    errors = 0

    def flush(rows: list[dict]) -> None:
        nonlocal write_header
        if not rows:
            return
        df = pd.DataFrame(rows)[OUTPUT_COLS]
        df.to_csv(OUTPUT_CSV, mode="a", index=False, header=write_header, encoding="utf-8")
        write_header = False

    # Processar do cache primeiro (rápido, sem API)
    print(f"  Processando {len(pending_from_cache)} do cache...")
    for fid in sorted(pending_from_cache):
        cp = _cache_path(fid)
        try:
            with open(cp, encoding="utf-8") as f:
                data = json.load(f)
            rows = extract_team_ratings(data, fid)
            all_rows.extend(rows)
        except Exception as e:
            errors += 1
        if len(all_rows) >= 200:
            flush(all_rows)
            all_rows = []

    flush(all_rows)
    all_rows = []

    # Buscar da API
    to_fetch_list = sorted(to_fetch)
    total_api = len(to_fetch_list)
    print(f"\n  Buscando {total_api} fixtures da API...")

    for i, fid in enumerate(to_fetch_list):
        if i % 50 == 0:
            pct = i * 100 // total_api if total_api else 100
            print(f"    {pct:3d}% ({i}/{total_api})  erros={errors}", end="\r", flush=True)

        try:
            data = api_get_players(fid)
            rows = extract_team_ratings(data, fid)
            all_rows.extend(rows)
            done_fids.add(fid)
        except Exception as e:
            errors += 1

        if len(all_rows) >= 200:
            flush(all_rows)
            all_rows = []

    flush(all_rows)
    print(f"\n    100% concluído.  Erros: {errors}")

    # Relatório final
    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        n_fixtures = df["match_id"].nunique()
        n_teams    = df["team"].nunique()
        print(f"\n  Cobertura final:")
        print(f"    Fixtures com ratings : {n_fixtures} / {len(all_fixture_ids)}")
        print(f"    Times únicos         : {n_teams}")
        if "rating_medio" in df.columns:
            print(f"    Rating médio global  : {df['rating_medio'].mean():.3f}")
            print(f"    Rating max global    : {df['rating_max'].mean():.3f}")
            # % de fixtures cobertos
            pct = n_fixtures / len(all_fixture_ids) * 100
            print(f"    Cobertura            : {pct:.1f}%")


if __name__ == "__main__":
    main()
