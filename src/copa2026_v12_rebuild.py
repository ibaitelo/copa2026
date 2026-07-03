#!/usr/bin/env python3
"""
copa2026_v12_rebuild.py — Re-executa seções 3-9 do v12 (após fix Czechia).
Roda rebuild + treino + standings + MC + Excel sem re-buscar da API.
"""
from __future__ import annotations
import pickle, random, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Importar todas as funções de copa2026_v12
from copa2026_v12 import (
    RAW_CSV, MODEL_CSV, ELO_CSV, OUT_PKL, OUT_XLSX,
    CUTOFF, SEED, N_SIM,
    FEATURES, XGB_PARAMS, CONF_DUMMIES, RATING_FEATURES,
    GROUPS, ALL_COPA_TEAMS, COPA_TEAM_CONF,
    R32, THIRD_ELIGIBLE, MATCH_VENUES,
    rebuild_model_dataset, apply_saves_normalization,
    wide_to_long_v12, train_v12, extract_team_feats, precompute_lambdas,
    compute_real_standings, best_third_place_real, assign_thirds,
    compute_r32_matchups, run_mc,
    generate_excel,
)
from xgboost import XGBRegressor

SEP = "=" * 72

def main():
    print(f"\n{SEP}")
    print("  COPA 2026 v12 — REBUILD (post Czechia fix)")
    print(f"{SEP}\n")

    raw = pd.read_csv(RAW_CSV)
    print(f"  Raw: {len(raw):,} linhas  |  WC: {(raw['match_type']=='WC').sum() // 2} jogos")

    # Rebuild model_dataset.csv
    print(f"\n{SEP}")
    print(f"  Rebuild model_dataset.csv (CUTOFF={CUTOFF.date()})")
    print(f"{SEP}")
    wide = rebuild_model_dataset(CUTOFF)
    print(f"  model_dataset: {len(wide):,} jogos wide")

    # Normalização de saves
    wide = apply_saves_normalization(wide)

    # Long format + treino
    print(f"\n{SEP}")
    print("  Treino xgboost_v12")
    print(f"{SEP}")
    long = wide_to_long_v12(wide)
    model = train_v12(long, CUTOFF)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"  Modelo salvo: {OUT_PKL}")

    # Features + lambdas
    team_feats = extract_team_feats(long)
    lam = precompute_lambdas(model, team_feats, ALL_COPA_TEAMS)
    print(f"  Lambdas: {len(lam)} pares")

    # Standings reais
    print(f"\n{SEP}")
    print("  Classificação real dos grupos")
    print(f"{SEP}")
    standings = compute_real_standings(raw)

    for g in sorted(GROUPS.keys()):
        df = standings[g]
        n_gms = int(df["pgj"].max())
        print(f"  Grupo {g}: {n_gms} jogos/time (máx)")
        for _, r in df.iterrows():
            print(f"    {r['pos']}. {r['team']:<25} "
                  f"{r['pts']}pts  {r['gf']}-{r['ga']} (SG{r['gd']:+})  "
                  f"({r['w']}V {r['d']}E {r['l']}D)")

    # R32 matchups
    print(f"\n{SEP}")
    print("  16 Avos de Final")
    print(f"{SEP}")
    matchups = compute_r32_matchups(standings, lam)
    for m in matchups:
        fav  = m["t1"] if m["pa1"] >= 0.5 else m["t2"]
        prob = max(m["pa1"], m["pa2"])
        print(f"  M{m['mid']:>2}  {m['t1']:<22} vs {m['t2']:<22}  → {fav} ({prob:.0%})")

    # Monte Carlo
    print(f"\n{SEP}")
    print(f"  Monte Carlo ({N_SIM:,} simulações)")
    print(f"{SEP}")
    champions, r32_adv = run_mc(lam, standings, N_SIM)
    total = sum(champions.values())
    print(f"\n  Top 20 Campeões:")
    for rank, (team, cnt) in enumerate(
        sorted(champions.items(), key=lambda x: x[1], reverse=True)[:20], 1
    ):
        pct = cnt / total * 100
        print(f"  {rank:>2}. {team:<25} {pct:5.1f}%")

    # Excel
    print(f"\n{SEP}")
    print("  Gerando Excel")
    print(f"{SEP}")
    generate_excel(matchups, standings, champions, r32_adv, N_SIM)

    print(f"\n{SEP}")
    print(f"  Concluído! Excel: {OUT_XLSX}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
