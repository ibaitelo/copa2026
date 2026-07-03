"""
compute_elo.py — ELO histórico para todas as seleções nacionais.
Fonte : data/raw/intl_results.csv (martj42/international_results)
Saída : data/raw/elo_history.csv  (date, team, elo_before, elo_after)

K factors (padrão WorldFootball ELO):
  Copa do Mundo final : 60   qualificatórias    : 40
  Finals continentais : 50   outros torneios     : 30
  Amistosos           : 20
Vantagem de casa: +100 pontos (campo neutro = 0)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INPUT_CSV   = Path("data/raw/intl_results.csv")
OUTPUT_CSV  = Path("data/raw/elo_history.csv")
START_YEAR  = 1990
INITIAL_ELO = 1500.0
HOME_ADV    = 100

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
    "Chinese Taipei":         "Chinese Taipei",
}

# Mapeamento de parte do nome do torneio → K factor
_K_RULES: list[tuple[str, int]] = [
    ("world cup",               60),
    ("copa do mundo",           60),
    ("confederations cup",      55),
    ("gold cup",                50),
    ("copa america",            50),
    ("copa américa",            50),
    ("african cup of nations",  50),
    ("africa cup of nations",   50),
    ("uefa euro",               50),
    ("afc asian cup",           50),
    ("olympics",                50),
    ("nations league",          45),
    ("concacaf championship",   40),
    ("qualifier",               40),
    ("qualification",           40),
    ("qualifying",              40),
    ("friendly",                20),
    ("international",           20),
]
_K_DEFAULT = 30


def _k(tournament: str) -> int:
    t = str(tournament).lower()
    for keyword, k in _K_RULES:
        if keyword in t:
            return k
    return _K_DEFAULT


def _expected(r1: float, r2: float, home_bonus: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r2 - r1 - home_bonus) / 400.0))


def main() -> None:
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df[df["date"].dt.year >= START_YEAR].copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["neutral"] = df["neutral"].map(
        {True: True, False: False, "TRUE": True, "FALSE": False, 1: True, 0: False}
    ).fillna(False)

    df["home_team"] = df["home_team"].map(lambda x: NAME_MAP.get(x, x))
    df["away_team"] = df["away_team"].map(lambda x: NAME_MAP.get(x, x))
    df = df.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = {}
    records: list[dict] = []

    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        neutral = bool(row["neutral"])
        k = _k(str(row.get("tournament", "Friendly")))
        date = row["date"]

        r_h = elo.get(ht, INITIAL_ELO)
        r_a = elo.get(at, INITIAL_ELO)
        bonus = 0.0 if neutral else HOME_ADV

        e_h = _expected(r_h, r_a, bonus)

        if hs > as_:
            w_h, w_a = 1.0, 0.0
        elif hs < as_:
            w_h, w_a = 0.0, 1.0
        else:
            w_h, w_a = 0.5, 0.5

        new_h = r_h + k * (w_h - e_h)
        new_a = r_a + k * (w_a - (1.0 - e_h))

        records.append({"date": date, "team": ht, "elo_before": round(r_h, 2), "elo_after": round(new_h, 2)})
        records.append({"date": date, "team": at, "elo_before": round(r_a, 2), "elo_after": round(new_a, 2)})

        elo[ht] = new_h
        elo[at] = new_a

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    n_games  = len(df)
    n_teams  = len(elo)
    n_rows   = len(df_out)
    print(f"ELO computado: {n_games:,} jogos, {n_teams} times, {n_rows:,} registros")
    print(f"Salvo: {OUTPUT_CSV}")

    top20 = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop-20 ELO atual (2025/2026):")
    for i, (team, r) in enumerate(top20, 1):
        print(f"  {i:2d}. {team:25s} {r:.1f}")


if __name__ == "__main__":
    main()
