#!/usr/bin/env python3
"""
validate_corrections.py — Correções cirúrgicas v10_clean
Sem reprocessar pipeline (build_features.py / integrate_datasets.py NÃO executados).

CORREÇÃO 1: Remove host_factor do modelo → xgboost_v10_clean.pkl
CORREÇÃO 2: host_factor como multiplicador pós-predição
            lambda_final = lambda_base × (1 + bonus)
            bonus = host_factor_cohost × 0.5  (co-sede no próprio país)
CORREÇÃO 3: distingue zero real de zero faltante em opp_saves_decay
            (salvaguarda — 0 casos no dataset atual, mas lógica aplicada)
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

# ─────────────────────────────────────────────────────────────────────────────
MODEL_DATASET   = Path("data/processed/model_dataset.csv")
HOST_FACTOR_CSV = Path("data/external/host_factor_copa2026.csv")
OUT_PKL         = Path("outputs/xgboost_v10_clean.pkl")
CUTOFF          = pd.Timestamp("2026-03-31")

HOST_TEAM_COUNTRY: dict[str, str] = {
    "United States": "USA",
    "Mexico":        "Mexico",
    "Canada":        "Canada",
}

CONF_DUMMIES = [
    "conf_AFC", "conf_CAF", "conf_CONCACAF",
    "conf_CONMEBOL", "conf_OFC", "conf_UEFA",
]
RATING_FEATURES = [
    "elo_diff_decay", "delta_rating_decay", "gls_ponderado_decay",
    "gls_decay_vs_forte", "gls_decay_vs_fraco", "margem_gols_decay",
]
# CORREÇÃO 1: host_factor removido do conjunto de features
BASE_FEATURES = ["opp_saves_decay", "shots_on_goal_decay", "gols_sofridos_decay"]
FEATURES      = BASE_FEATURES + RATING_FEATURES + CONF_DUMMIES  # 15 features

XGB_PARAMS = dict(
    objective        = "count:poisson",
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)


# ─────────────────────────────────────────────────────────────────────────────
# CORREÇÃO 2 — host_factor como multiplicador pós-predição
# ─────────────────────────────────────────────────────────────────────────────
def load_cohost_factors() -> dict[str, float]:
    df = pd.read_csv(HOST_FACTOR_CSV)
    return {row["team"]: float(row["host_factor_cohost"]) for _, row in df.iterrows()}


def get_home_bonus(team: str, venue_country: str, hf_cohost: dict[str, float]) -> float:
    """bonus = host_factor_cohost × 0.5 apenas quando co-sede joga no PRÓPRIO país."""
    tc = HOST_TEAM_COUNTRY.get(team)
    if tc is None or tc != venue_country:
        return 0.0
    return hf_cohost.get(team, 0.0) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# CORREÇÃO 3 — opp_saves_decay: distingue zero real de zero faltante
# ─────────────────────────────────────────────────────────────────────────────
def compute_conf_medians(wide: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Calcula medianas de saves_decay por confederação usando dados reais (não-imputed, >0)."""
    real_vals: list[float] = []
    for side in ["home", "away"]:
        col = f"{side}_saves_decay"
        imp = f"{side}_saves_decay_imputed"
        if col not in wide.columns:
            continue
        imp_mask = wide[imp].astype(bool) if imp in wide.columns else pd.Series(False, index=wide.index)
        real_vals.extend(wide.loc[~imp_mask & (wide[col] > 0), col].tolist())

    global_median = float(np.median(real_vals)) if real_vals else 2.5

    conf_medians: dict[str, float] = {}
    for conf in CONF_DUMMIES:
        name = conf.replace("conf_", "")
        v: list[float] = []
        for side in ["home"]:  # conf_ é game-level baseado no time home
            col = f"{side}_saves_decay"
            imp = f"{side}_saves_decay_imputed"
            if col not in wide.columns or conf not in wide.columns:
                continue
            imp_mask = wide[imp].astype(bool) if imp in wide.columns else pd.Series(False, index=wide.index)
            v.extend(wide.loc[wide[conf].astype(bool) & ~imp_mask & (wide[col] > 0), col].tolist())
        conf_medians[name] = float(np.median(v)) if v else global_median

    return global_median, conf_medians


def apply_saves_correction(wide: pd.DataFrame,
                           global_median: float,
                           conf_medians: dict[str, float]) -> pd.DataFrame:
    """
    Cria {side}_saves_decay_fixed para home e away.
    Regras (por ordem de prioridade):
      1. imputed=True          → manter valor atual (já é mediana conf aplicada)
      2. saves=0, has_data=True  → manter 0 (zero real de goleiro sem defesas)
      3. saves=0, has_data=False → substituir pela mediana da confederação
    """
    wide = wide.copy()

    # fallback por linha baseado nas colunas conf_ (game-level = conf do home)
    fallback = np.full(len(wide), global_median)
    for conf in CONF_DUMMIES:
        name = conf.replace("conf_", "")
        if conf in wide.columns:
            fallback = np.where(wide[conf].astype(bool), conf_medians.get(name, global_median), fallback)

    for side in ["home", "away"]:
        saves_col    = f"{side}_saves_decay"
        imputed_col  = f"{side}_saves_decay_imputed"
        has_data_col = f"{side}_has_decay_data"
        fixed_col    = f"{side}_saves_decay_fixed"

        if saves_col not in wide.columns:
            wide[fixed_col] = global_median
            continue

        fixed = wide[saves_col].astype(float).copy()

        imputed_mask  = (wide[imputed_col].astype(bool)
                         if imputed_col in wide.columns
                         else pd.Series(False, index=wide.index))
        has_data_mask = (wide[has_data_col].astype(bool)
                         if has_data_col in wide.columns
                         else pd.Series(True, index=wide.index))

        # Caso 3: zero + sem dados reais + não-imputed → substituir
        zero_missing = (~imputed_mask) & (fixed == 0) & (~has_data_mask)
        fixed[zero_missing] = pd.Series(fallback, index=wide.index)[zero_missing]

        wide[fixed_col] = fixed

    return wide


# ─────────────────────────────────────────────────────────────────────────────
# Wide → Long (clean: sem host_factor, com saves corrigido)
# ─────────────────────────────────────────────────────────────────────────────
def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for side, opp in [("home", "away"), ("away", "home")]:
            opp_saves = float(r.get(f"{opp}_saves_decay_fixed",
                                    r.get(f"{opp}_saves_decay", 2.5)))
            own_saves = float(r.get(f"{side}_saves_decay_fixed",
                                    r.get(f"{side}_saves_decay", 2.5)))
            row: dict = {
                "date":                r["date"],
                "team":                r[f"{side}_team"],
                "gols_marcados":       r[f"{side}_gols"],
                "opp_saves_decay":     opp_saves,
                "shots_on_goal_decay": r[f"{side}_shots_on_goal_decay"],
                "gols_sofridos_decay": r.get(f"{side}_gols_sofridos_decay", np.nan),
                "saves_decay":         own_saves,   # armazenado em team_feats
            }
            for col in CONF_DUMMIES:
                row[col] = int(r.get(col, 0))
            for col in RATING_FEATURES:
                raw = r.get(f"{side}_{col}", np.nan)
                row[col] = float(raw) if pd.notna(raw) else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Treino
# ─────────────────────────────────────────────────────────────────────────────
def train_clean(long: pd.DataFrame) -> XGBRegressor:
    train = long[long["date"] <= CUTOFF]
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURES], train["gols_marcados"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Extração de team_feats
# ─────────────────────────────────────────────────────────────────────────────
def extract_team_feats(long: pd.DataFrame) -> dict[str, dict]:
    team_feats: dict[str, dict] = {}
    for _, row in long.sort_values("date").iterrows():
        entry: dict = {
            "saves_decay":         float(row.get("saves_decay", 2.5)),
            "shots_on_goal_decay": float(row.get("shots_on_goal_decay", 3.5)),
            "gols_sofridos_decay": row.get("gols_sofridos_decay", np.nan),
        }
        for col in RATING_FEATURES:
            v = row.get(col, np.nan)
            entry[col] = float(v) if pd.notna(v) else np.nan
        team_feats[row["team"]] = entry

    elo_path = Path("data/raw/elo_history.csv")
    if elo_path.exists():
        elo_hist = pd.read_csv(elo_path, parse_dates=["date"])
        for team, elo_val in (elo_hist.sort_values("date")
                              .groupby("team")["elo_after"].last()
                              .items()):
            tf = team_feats.setdefault(team, {})
            tf["_elo_current"] = float(elo_val)

    return team_feats


# ─────────────────────────────────────────────────────────────────────────────
# Predição de 1 jogo (lambda_base + multiplicador)
# ─────────────────────────────────────────────────────────────────────────────
def predict_game(model: XGBRegressor,
                 team_feats: dict[str, dict],
                 t1: str, t2: str,
                 venue_country: str,
                 hf_cohost: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    rows = []
    for team, opp in [(t1, t2), (t2, t1)]:
        f1 = team_feats.get(team, {})
        f2 = team_feats.get(opp, {})
        row: dict = {
            "opp_saves_decay":     f2.get("saves_decay", 2.5),
            "shots_on_goal_decay": f1.get("shots_on_goal_decay", 3.5),
            "gols_sofridos_decay": f1.get("gols_sofridos_decay", np.nan),
            **{c: 0 for c in CONF_DUMMIES},
        }
        for col in RATING_FEATURES:
            if col == "elo_diff_decay":
                e1 = f1.get("_elo_current", np.nan)
                e2 = f2.get("_elo_current", np.nan)
                row[col] = (float(e1) - float(e2)
                            if pd.notna(e1) and pd.notna(e2)
                            else f1.get(col, np.nan))
            else:
                row[col] = f1.get(col, np.nan)
        rows.append(row)

    lams   = model.predict(pd.DataFrame(rows)[FEATURES])
    lh_b   = float(lams[0])
    la_b   = float(lams[1])
    bonus_h = get_home_bonus(t1, venue_country, hf_cohost)
    bonus_a = get_home_bonus(t2, venue_country, hf_cohost)
    return lh_b, lh_b * (1 + bonus_h), la_b, la_b * (1 + bonus_a), bonus_h, bonus_a


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    SEP  = "=" * 72
    SEP2 = "-" * 72

    print(f"\n{SEP}")
    print("  CORREÇÕES CIRÚRGICAS — Copa 2026 v10_clean")
    print(f"  (sem build_features.py / integrate_datasets.py)")
    print(f"{SEP}\n")

    # ── Carregar dataset ──────────────────────────────────────────────────────
    wide = pd.read_csv(MODEL_DATASET, parse_dates=["date"])
    print(f"  Dataset: {len(wide)} jogos × {len(wide.columns)} colunas")

    # ── CORREÇÃO 3 — diagnóstico e aplicação ──────────────────────────────────
    print(f"\n{SEP}")
    print("  CORREÇÃO 3 — Diagnóstico: zeros reais vs faltantes em saves_decay")
    print(f"{SEP}")
    total_fixed = 0
    for side in ["home", "away"]:
        saves   = wide[f"{side}_saves_decay"]
        imputed = wide[f"{side}_saves_decay_imputed"].astype(bool)
        has_data = wide[f"{side}_has_decay_data"].astype(bool)
        c1 = imputed.sum()
        c2 = (~imputed & (saves == 0) & has_data).sum()
        c3 = (~imputed & (saves == 0) & ~has_data).sum()
        total_fixed += c3
        print(f"  {side.upper()}:")
        print(f"    imputed=True  (mediana conf já aplicada, manter): {c1}")
        print(f"    zero + has_data=True  (zero REAL, manter):        {c2}")
        print(f"    zero + has_data=False (zero FALTANTE, FIXAR):    {c3}")

    global_median, conf_medians = compute_conf_medians(wide)
    print(f"\n  Mediana global de fallback: {global_median:.4f}")
    print(f"  Casos corrigidos: {total_fixed} {'✓ (nenhum — dataset limpo)' if total_fixed == 0 else '← FIXADOS'}")

    wide_fixed = apply_saves_correction(wide, global_median, conf_medians)

    # ── Antes/depois por time ─────────────────────────────────────────────────
    print(f"\n  saves_decay por time (= opp_saves_decay quando enfrentados):")
    print(f"  {'Time':20s}  {'antes':>10}  {'depois':>11}  {'Δ':>8}  {'zeros→fix':>10}")
    print("  " + SEP2)

    teams_check = ["Brazil", "Mexico", "Panama", "Portugal", "Austria"]
    for team in teams_check:
        b_vals, a_vals, zf = [], [], 0
        for side in ["home", "away"]:
            mask = wide_fixed[f"{side}_team"] == team
            col_b = f"{side}_saves_decay"
            col_a = f"{side}_saves_decay_fixed"
            if col_b in wide_fixed.columns:
                bv = wide_fixed.loc[mask, col_b].tolist()
                av = wide_fixed.loc[mask, col_a].tolist() if col_a in wide_fixed.columns else bv
                b_vals.extend(bv)
                a_vals.extend(av)
                zf += sum(1 for x, y in zip(bv, av) if x == 0 and y != 0)
        mb = np.mean(b_vals) if b_vals else np.nan
        ma = np.mean(a_vals) if a_vals else np.nan
        d  = ma - mb if pd.notna(mb) and pd.notna(ma) else np.nan
        print(f"  {team:20s}  {mb:10.4f}  {ma:11.4f}  {d:+8.4f}  {zf:>10d}")

    # ── Construir long e treinar ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  CORREÇÃO 1 — Treinando xgboost_v10_clean.pkl (sem host_factor)")
    print(f"  Features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"{SEP}")

    long = wide_to_long(wide_fixed)
    model = train_clean(long)

    with open(OUT_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"  Modelo salvo: {OUT_PKL}")

    # ── VALIDAÇÃO 1: confirmar features ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  VALIDAÇÃO 1 — Features do xgboost_v10_clean.pkl")
    print(f"{SEP}")
    for i, feat in enumerate(FEATURES, 1):
        print(f"    {i:2d}. {feat}")
    assert "host_factor" not in FEATURES, "ERRO: host_factor ainda presente!"
    print(f"\n  ✓ host_factor AUSENTE ({len(FEATURES)} features)\n")

    # ── Extrair team_feats ────────────────────────────────────────────────────
    team_feats = extract_team_feats(long)
    hf_cohost  = load_cohost_factors()

    # Mostrar bonus esperados
    print(f"  CORREÇÃO 2 — bonus pós-predição (host_factor_cohost × 0.5):")
    for team, country in HOST_TEAM_COUNTRY.items():
        bonus = hf_cohost.get(team, 0.0) * 0.5
        print(f"    {team:20s} jogando em {country:8s} → bonus = {bonus:.4f}"
              f"  (cohost={hf_cohost.get(team, 0):.4f})")

    # ── VALIDAÇÃO 2: 5 jogos ─────────────────────────────────────────────────
    games = [
        ("Mexico",        "South Africa", "Cidade do México", "Mexico"),
        ("United States", "Paraguay",     "Los Angeles",      "USA"),
        ("Brazil",        "Morocco",      "Houston",          "USA"),
        ("Panama",        "Croatia",      "Toronto",          "Canada"),
        ("Austria",       "Jordan",       "San Francisco",    "USA"),
    ]

    print(f"\n{SEP}")
    print("  VALIDAÇÃO 2 — Lambdas por jogo")
    print(f"  λ_final = λ_base × (1 + bonus)   |   bonus para co-sede no próprio país")
    print(f"{SEP}")
    print(f"\n  {'Jogo':33s}  {'Venue':17s}"
          f"  {'λH_base':>8} {'λH_fin':>8} {'bonH':>6}"
          f"  {'λA_base':>8} {'λA_fin':>8} {'bonA':>6}")
    print("  " + "-" * 103)

    for t1, t2, city, venue_country in games:
        lh_b, lh_f, la_b, la_f, bh, ba = predict_game(
            model, team_feats, t1, t2, venue_country, hf_cohost
        )
        label = f"{t1} × {t2}"
        print(f"  {label:33s}  {city:17s}"
              f"  {lh_b:8.4f} {lh_f:8.4f} {bh:6.4f}"
              f"  {la_b:8.4f} {la_f:8.4f} {ba:6.4f}")

    print(f"\n{SEP}")
    print("  ✓ CONCLUÍDO — outputs/xgboost_v10_clean.pkl salvo")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
