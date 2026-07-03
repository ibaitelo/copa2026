"""
compute_starter_ratings.py — Rating médio dos titulares por partida.

Para cada JSON em data/raw/api_cache/fixtures_players/{id}.json, extrai o
rating médio dos N_STARTERS jogadores com mais minutos (excluindo entradas
tardias com ≤ MIN_MINUTES minutos).

Saída: data/raw/starter_ratings_per_game.csv (match_id, team, rating_titulares, n_players)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE_DIR  = Path("data/raw/api_cache/fixtures_players")
OUTPUT_CSV = Path("data/raw/starter_ratings_per_game.csv")
N_STARTERS = 11
MIN_MINUTES = 10   # exclui entradas tardias (últimos ≤10 min)
MIN_PLAYERS = 5    # mínimo de jogadores elegíveis para cálculo confiável

NAME_MAP: dict[str, str] = {
    "USA":                    "United States",
    "Korea Republic":         "South Korea",
    "Korea DPR":              "North Korea",
    "IR Iran":                "Iran",
    "FYR Macedonia":          "North Macedonia",
    "Bosnia & Herzegovina":   "Bosnia and Herzegovina",
    "Türkiye":                "Turkey",
    "Cape Verde Islands":     "Cape Verde",
    "Congo DR":               "DR Congo",
    "Cote d'Ivoire":          "Ivory Coast",
    "Côte d'Ivoire":          "Ivory Coast",
    "México":                 "Mexico",
    "Curaçao":                "Curacao",
}


def _process_fixture(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    records: list[dict] = []
    for team_data in data.get("response", []):
        team_raw  = team_data.get("team", {}).get("name", "")
        team_name = NAME_MAP.get(team_raw, team_raw)
        players   = team_data.get("players", [])

        eligible: list[dict] = []
        for p in players:
            stats_list = p.get("statistics", [{}])
            if not stats_list:
                continue
            stats   = stats_list[0]
            games   = stats.get("games", {})
            minutes = games.get("minutes")
            rating  = games.get("rating")
            if minutes is None or rating is None:
                continue
            try:
                minutes = int(minutes)
                rating  = float(rating)
            except (ValueError, TypeError):
                continue
            if minutes > MIN_MINUTES:
                eligible.append({"minutes": minutes, "rating": rating})

        if len(eligible) < MIN_PLAYERS:
            continue

        eligible.sort(key=lambda x: x["minutes"], reverse=True)
        top_n = eligible[:N_STARTERS]
        avg_r = float(np.mean([p["rating"] for p in top_n]))

        records.append({
            "match_id":         path.stem,
            "team":             team_name,
            "rating_titulares": round(avg_r, 4),
            "n_players":        len(top_n),
        })

    return records


def main() -> None:
    paths = sorted(CACHE_DIR.glob("*.json"))
    if not paths:
        print(f"AVISO: Nenhum JSON encontrado em {CACHE_DIR}")
        return

    print(f"Processando {len(paths)} arquivos de ratings de jogadores...")
    all_records: list[dict] = []
    errors = 0

    for path in paths:
        try:
            records = _process_fixture(path)
            all_records.extend(records)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Erro em {path.name}: {e}")

    df = pd.DataFrame(all_records)
    if df.empty:
        print("AVISO: nenhum dado processado!")
        return

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"Starter ratings: {len(df)} linhas, {df['match_id'].nunique()} partidas, {errors} erros")
    print(f"Salvo: {OUTPUT_CSV}")
    print(f"\nEstatísticas rating_titulares:")
    print(df["rating_titulares"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
