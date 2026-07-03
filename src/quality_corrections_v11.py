#!/usr/bin/env python3
"""
quality_corrections_v11.py — Correções de qualidade no dataset

CORREÇÃO 1 — Normalização de nomes inconsistentes:
  Congo               → DR Congo
  Bosnia-Herzegovina  → Bosnia and Herzegovina
  Equ. Guinea         → Equatorial Guinea
  (Korea Republic e IR Iran já mapeados em NAME_MAP)

CORREÇÃO 2 — home_advantage em amistosos:
  Todos match_type='friendly' → home_advantage=0

CORREÇÃO 3 — Verificação Levenshtein:
  Times Copa 2026 com variações próximas (informativo)

Regenera model_dataset.csv, retreina xgboost_v11.pkl, MC 2.000.
Critério: DR Congo < 2.5%
"""
from __future__ import annotations

import pickle
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_CSV         = Path("data/raw/full_dataset_raw.csv")
MODEL_DATASET   = Path("data/processed/model_dataset.csv")
HOST_FACTOR_CSV = Path("data/external/host_factor_copa2026.csv")
ELO_CSV         = Path("data/raw/elo_history.csv")
OUT_PKL         = Path("outputs/xgboost_v11.pkl")
CUTOFF          = pd.Timestamp("2026-03-31")
SEED            = 42

# ── Mapeamentos de qualidade ──────────────────────────────────────────────────
NAME_CORRECTIONS = {
    "Congo":              "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Equ. Guinea":        "Equatorial Guinea",
    "Korea Republic":     "South Korea",
    "IR Iran":            "Iran",
}

COPA2026_TEAMS = {
    "Argentina","Brazil","Colombia","Ecuador","Uruguay","Paraguay",
    "Germany","France","Spain","England","Netherlands","Portugal",
    "Belgium","Austria","Switzerland","Croatia","Norway","Sweden",
    "Czech Republic","Turkey","Scotland","Bosnia and Herzegovina",
    "Japan","South Korea","Iran","Australia","Saudi Arabia",
    "Uzbekistan","Jordan","Iraq","Qatar",
    "Morocco","Senegal","Egypt","Ghana","Cape Verde",
    "DR Congo","Ivory Coast","South Africa","Algeria","Tunisia",
    "United States","Mexico","Canada","Panama","Curacao","Haiti",
    "New Zealand",
}

COPA_TEAM_CONF = {
    "Argentina":"CONMEBOL","Brazil":"CONMEBOL","Colombia":"CONMEBOL",
    "Ecuador":"CONMEBOL","Uruguay":"CONMEBOL","Paraguay":"CONMEBOL",
    "Germany":"UEFA","France":"UEFA","Spain":"UEFA","England":"UEFA",
    "Netherlands":"UEFA","Portugal":"UEFA","Belgium":"UEFA","Austria":"UEFA",
    "Switzerland":"UEFA","Croatia":"UEFA","Norway":"UEFA","Sweden":"UEFA",
    "Czech Republic":"UEFA","Turkey":"UEFA","Scotland":"UEFA",
    "Bosnia and Herzegovina":"UEFA",
    "Japan":"AFC","South Korea":"AFC","Iran":"AFC","Australia":"AFC",
    "Saudi Arabia":"AFC","Uzbekistan":"AFC","Jordan":"AFC","Iraq":"AFC",
    "Qatar":"AFC",
    "Morocco":"CAF","Senegal":"CAF","Egypt":"CAF","Ghana":"CAF",
    "Cape Verde":"CAF","DR Congo":"CAF","Ivory Coast":"CAF",
    "South Africa":"CAF","Algeria":"CAF","Tunisia":"CAF",
    "United States":"CONCACAF","Mexico":"CONCACAF","Canada":"CONCACAF",
    "Panama":"CONCACAF","Curacao":"CONCACAF","Haiti":"CONCACAF",
    "New Zealand":"OFC",
}

CONF_DUMMIES = [
    "conf_AFC","conf_CAF","conf_CONCACAF","conf_CONMEBOL","conf_OFC","conf_UEFA",
]
RATING_FEATURES = [
    "elo_diff_decay","delta_rating_decay","gls_ponderado_decay",
    "gls_decay_vs_forte","gls_decay_vs_fraco","margem_gols_decay",
]
FEATURES = (
    ["opp_saves_decay","shots_on_goal_decay","gols_sofridos_decay"]
    + RATING_FEATURES + CONF_DUMMIES
)
XGB_PARAMS = dict(
    objective="count:poisson", n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, random_state=42, n_jobs=-1, verbosity=0,
)
HOST_TEAM_COUNTRY = {
    "United States": "USA", "Mexico": "Mexico", "Canada": "Canada",
}


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — Levenshtein (informativo)
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]


def check_levenshtein(raw: pd.DataFrame) -> None:
    all_teams = (set(raw["team"].dropna().unique())
                 | set(raw["opponent"].dropna().unique()))
    print(f"  Times únicos no raw: {len(all_teams)}")
    suspicious: list[tuple] = []
    for copa_t in sorted(COPA2026_TEAMS):
        for t in sorted(all_teams):
            if t != copa_t and t not in COPA2026_TEAMS:
                d = levenshtein(copa_t.lower(), t.lower())
                if d <= 2:  # apenas distância ≤ 2 (mais restrito que ≤ 3)
                    n = raw[raw["team"] == t]["match_id"].nunique()
                    suspicious.append((copa_t, t, d, n))
    if suspicious:
        print("  Possíveis duplicatas (Levenshtein ≤ 2):")
        for copa_t, variant, d, n in sorted(suspicious, key=lambda x: x[2]):
            print(f"    {copa_t:<28} ~ {variant:<28} (d={d}, {n} jogos)")
    else:
        print("  Nenhuma duplicata suspeita encontrada.")


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — Correções no raw CSV
# ─────────────────────────────────────────────────────────────────────────────

def apply_name_corrections(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = {name: 0 for name in NAME_CORRECTIONS}
    for col in ["team", "opponent"]:
        for old, new in NAME_CORRECTIONS.items():
            mask = raw[col] == old
            if mask.any():
                counts[old] += int(mask.sum())
                raw.loc[mask, col] = new
    return raw, counts


def apply_friendly_home_advantage(raw: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    friendly_mask = raw["match_type"] == "friendly"
    home_mask = friendly_mask & (raw["home_away"] == "home")
    away_mask = friendly_mask & (raw["home_away"] == "away")

    mean_home_before = raw.loc[home_mask, "gols_marcados"].mean()
    mean_away_before = raw.loc[away_mask, "gols_marcados"].mean()

    n_changed = int((friendly_mask & (raw["home_advantage"] != 0)).sum())
    raw.loc[friendly_mask, "home_advantage"] = 0

    mean_home_after = raw.loc[home_mask, "gols_marcados"].mean()
    mean_away_after = raw.loc[away_mask, "gols_marcados"].mean()

    return raw, mean_home_before, mean_home_after, mean_away_before, mean_away_after, n_changed


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 — Rebuild features + saves correction + normalization
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_model_dataset():
    from build_features import (
        load_and_merge_raw, add_decay_features, impute_by_confederation,
        build_match_dataset, XI_DEFAULT, save_data,
    )
    df_raw  = load_and_merge_raw()
    df_dec  = add_decay_features(df_raw, XI_DEFAULT)
    df_imp  = impute_by_confederation(df_dec)
    df_wide = build_match_dataset(df_imp)
    save_data(df_wide)
    return df_wide


def apply_saves_correction(wide: pd.DataFrame) -> tuple[pd.DataFrame, float, dict]:
    real_vals: list[float] = []
    for side in ["home", "away"]:
        col, imp = f"{side}_saves_decay", f"{side}_saves_decay_imputed"
        if col not in wide.columns:
            continue
        imp_m = wide[imp].astype(bool) if imp in wide.columns else pd.Series(False, index=wide.index)
        real_vals.extend(wide.loc[~imp_m & (wide[col] > 0), col].tolist())
    global_med = float(np.median(real_vals)) if real_vals else 2.5

    conf_med: dict[str, float] = {}
    for conf in CONF_DUMMIES:
        name = conf.replace("conf_", "")
        col, imp = "home_saves_decay", "home_saves_decay_imputed"
        if col in wide.columns and conf in wide.columns:
            imp_m = wide[imp].astype(bool) if imp in wide.columns else pd.Series(False, index=wide.index)
            v = wide.loc[wide[conf].astype(bool) & ~imp_m & (wide[col] > 0), col].tolist()
            conf_med[name] = float(np.median(v)) if v else global_med

    fallback = np.full(len(wide), global_med)
    for conf in CONF_DUMMIES:
        name = conf.replace("conf_", "")
        if conf in wide.columns:
            fallback = np.where(wide[conf].astype(bool), conf_med.get(name, global_med), fallback)

    wide = wide.copy()
    for side in ["home", "away"]:
        saves_col, imp_col = f"{side}_saves_decay", f"{side}_saves_decay_imputed"
        has_col, fixed_col = f"{side}_has_decay_data", f"{side}_saves_decay_fixed"
        if saves_col not in wide.columns:
            wide[fixed_col] = global_med; continue
        fixed = wide[saves_col].astype(float).copy()
        imp_m = wide[imp_col].astype(bool) if imp_col in wide.columns else pd.Series(False, index=wide.index)
        has_m = wide[has_col].astype(bool)  if has_col in wide.columns else pd.Series(True,  index=wide.index)
        fixed[(~imp_m) & (fixed == 0) & (~has_m)] = pd.Series(fallback, index=wide.index)[(~imp_m) & (fixed == 0) & (~has_m)]
        wide[fixed_col] = fixed
    return wide, global_med, conf_med


def compute_conf_medians_by_team(wide: pd.DataFrame, team_conf: dict) -> tuple[float, dict]:
    all_saves: list[pd.DataFrame] = []
    for side in ["home", "away"]:
        saves_col = f"{side}_saves_decay_fixed"
        if saves_col not in wide.columns:
            saves_col = f"{side}_saves_decay"
        imp_col  = f"{side}_saves_decay_imputed"
        team_col = f"{side}_team"
        if saves_col not in wide.columns:
            continue
        imp_m  = wide[imp_col].astype(bool) if imp_col in wide.columns else pd.Series(False, index=wide.index)
        real_m = ~imp_m & (wide[saves_col] > 0)
        sub = wide.loc[real_m, [team_col, saves_col]].copy()
        sub.columns = ["team", "saves"]
        sub["conf"] = sub["team"].map(team_conf)
        all_saves.append(sub)
    if not all_saves:
        return 2.5, {}
    combined = pd.concat(all_saves, ignore_index=True)
    return float(combined["saves"].median()), combined.groupby("conf")["saves"].median().to_dict()


def wide_to_long_v11(df: pd.DataFrame, team_conf: dict,
                     conf_medians: dict, global_med: float) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in df.iterrows():
        for side, opp in [("home", "away"), ("away", "home")]:
            opp_team = r[f"{opp}_team"]
            own_team = r[f"{side}_team"]
            opp_raw  = float(r.get(f"{opp}_saves_decay_fixed", r.get(f"{opp}_saves_decay", 2.5)))
            own_raw  = float(r.get(f"{side}_saves_decay_fixed", r.get(f"{side}_saves_decay", 2.5)))
            opp_imp  = bool(r.get(f"{opp}_saves_decay_imputed", False))
            own_imp  = bool(r.get(f"{side}_saves_decay_imputed", False))

            def _norm(raw_v, team, imp):
                if imp: return 1.0
                c = team_conf.get(team, "UEFA")
                m = conf_medians.get(c, global_med)
                return raw_v / m if m > 0 else 1.0

            row: dict = {
                "date":                r["date"],
                "team":                own_team,
                "gols_marcados":       r[f"{side}_gols"],
                "opp_saves_decay":     _norm(opp_raw, opp_team, opp_imp),
                "shots_on_goal_decay": r[f"{side}_shots_on_goal_decay"],
                "gols_sofridos_decay": r.get(f"{side}_gols_sofridos_decay", np.nan),
                "saves_decay":         own_raw,
                "saves_decay_norm":    _norm(own_raw, own_team, own_imp),
            }
            for col in CONF_DUMMIES:
                row[col] = int(r.get(col, 0))
            for col in RATING_FEATURES:
                raw_v = r.get(f"{side}_{col}", np.nan)
                row[col] = float(raw_v) if pd.notna(raw_v) else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — Treino e MC
# ─────────────────────────────────────────────────────────────────────────────

def train_v11(long: pd.DataFrame) -> XGBRegressor:
    train = long[long["date"] <= CUTOFF]
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURES], train["gols_marcados"])
    return model


def extract_team_feats(long: pd.DataFrame) -> dict:
    team_feats: dict = {}
    for _, row in long.sort_values("date").iterrows():
        entry = {
            "saves_decay":         float(row.get("saves_decay_norm", row.get("saves_decay", 2.5))),
            "saves_decay_raw":     float(row.get("saves_decay", 2.5)),
            "shots_on_goal_decay": float(row.get("shots_on_goal_decay", 3.5)),
            "gols_sofridos_decay": row.get("gols_sofridos_decay", np.nan),
        }
        for col in RATING_FEATURES:
            v = row.get(col, np.nan)
            entry[col] = float(v) if pd.notna(v) else np.nan
        team_feats[row["team"]] = entry
    if ELO_CSV.exists():
        elo = pd.read_csv(ELO_CSV, parse_dates=["date"])
        for team, val in elo.sort_values("date").groupby("team")["elo_after"].last().items():
            team_feats.setdefault(team, {})["_elo_current"] = float(val)
    return team_feats


def precompute_lambdas(model, team_feats, teams):
    rows, pairs = [], []
    for t1 in teams:
        for t2 in teams:
            if t1 == t2: continue
            f1, f2 = team_feats.get(t1, {}), team_feats.get(t2, {})
            row: dict = {
                "opp_saves_decay":     f2.get("saves_decay", 1.0),
                "shots_on_goal_decay": f1.get("shots_on_goal_decay", 3.5),
                "gols_sofridos_decay": f1.get("gols_sofridos_decay", np.nan),
                **{c: 0 for c in CONF_DUMMIES},
            }
            for col in RATING_FEATURES:
                if col == "elo_diff_decay":
                    e1, e2 = f1.get("_elo_current", np.nan), f2.get("_elo_current", np.nan)
                    row[col] = float(e1) - float(e2) if pd.notna(e1) and pd.notna(e2) else f1.get(col, np.nan)
                else:
                    row[col] = f1.get(col, np.nan)
            rows.append(row); pairs.append((t1, t2))
    preds = model.predict(pd.DataFrame(rows)[FEATURES])
    return {pair: float(p) for pair, p in zip(pairs, preds)}


def load_cohost_factors():
    df = pd.read_csv(HOST_FACTOR_CSV)
    return {row["team"]: float(row["host_factor_cohost"]) for _, row in df.iterrows()}


def get_match_lambdas(t1, t2, venue, lam, hf):
    def _bonus(team, v):
        tc = HOST_TEAM_COUNTRY.get(team)
        return hf.get(team, 0.0) * 0.5 if (tc and tc == v) else 0.0
    return (lam.get((t1, t2), 1.2) * (1 + _bonus(t1, venue)),
            lam.get((t2, t1), 1.0) * (1 + _bonus(t2, venue)))


def run_mc_2000(lam, hf, n_sim=2000):
    from simulate_copa2026 import (
        GROUPS, R32, MATCH_VENUES, GROUP_VENUE_COUNTRY,
        assign_thirds, simulate_game_from_lambdas,
    )
    random.seed(SEED); np.random.seed(SEED)
    champions: dict[str, int] = {}

    def sim_group(gid, teams):
        venue = GROUP_VENUE_COUNTRY[gid]
        stats = {t: {"pts":0,"gf":0,"ga":0,"gd":0} for t in teams}
        for t1, t2 in combinations(teams, 2):
            lh, la = get_match_lambdas(t1, t2, venue, lam, hf)
            g1, g2 = simulate_game_from_lambdas(lh, la, stochastic=True)
            if g1 > g2: stats[t1]["pts"] += 3
            elif g2 > g1: stats[t2]["pts"] += 3
            else: stats[t1]["pts"] += 1; stats[t2]["pts"] += 1
            for t, gf, ga in [(t1,g1,g2),(t2,g2,g1)]:
                stats[t]["gf"] += gf; stats[t]["ga"] += ga; stats[t]["gd"] += gf - ga
        df = pd.DataFrame([{"team":t,**v} for t,v in stats.items()])
        df = df.sort_values(["pts","gd","gf"], ascending=False).reset_index(drop=True)
        df["pos"] = df.index + 1
        return df

    def best_thirds(standings):
        thirds = [{"group":g,"team":df[df["pos"]==3].iloc[0]["team"],
                   "pts":df[df["pos"]==3].iloc[0]["pts"],
                   "gd":df[df["pos"]==3].iloc[0]["gd"],
                   "gf":df[df["pos"]==3].iloc[0]["gf"],
                   "_tb":random.random()}
                  for g, df in standings.items()]
        thirds.sort(key=lambda x: (x["pts"],x["gd"],x["gf"],x["_tb"]), reverse=True)
        return thirds[:8]

    def sim_ko(t1, t2, mid):
        venue = MATCH_VENUES.get(mid, "USA")
        lh, la = get_match_lambdas(t1, t2, venue, lam, hf)
        g1, g2 = int(np.random.poisson(lh)), int(np.random.poisson(la))
        if g1 != g2: return t1 if g1 > g2 else t2
        g1 += int(np.random.poisson(lh/3)); g2 += int(np.random.poisson(la/3))
        if g1 != g2: return t1 if g1 > g2 else t2
        return t1 if random.random() < 0.5 else t2

    def run_ko(standings, q_thirds):
        t_by_grp = {t["group"]: t["team"] for t in q_thirds}
        t_asgn   = assign_thirds(q_thirds)
        def pos(g, n): return standings[g][standings[g]["pos"]==n].iloc[0]["team"]
        def resolve(slot, mid):
            if slot[0]=="1": return pos(slot[1], 1)
            if slot[0]=="2": return pos(slot[1], 2)
            g = t_asgn.get(mid); return t_by_grp.get(g, next(iter(t_by_grp.values())))
        mw = {}
        for mid, a_slot, b_slot, _ in R32:
            mw[mid] = sim_ko(resolve(a_slot, mid), resolve(b_slot, mid), mid)
        for mid, m1, m2 in [(89,73,74),(90,75,76),(91,77,78),(92,79,80),
                             (93,81,82),(94,83,84),(95,85,86),(96,87,88)]:
            mw[mid] = sim_ko(mw[m1], mw[m2], mid)
        for mid, m1, m2 in [(97,89,90),(98,91,92),(99,93,94),(100,95,96)]:
            mw[mid] = sim_ko(mw[m1], mw[m2], mid)
        sf_los = {}
        for mid, m1, m2 in [(101,97,98),(102,99,100)]:
            t1, t2 = mw[m1], mw[m2]; w = sim_ko(t1, t2, mid)
            mw[mid] = w; sf_los[mid] = t2 if w == t1 else t1
        mw[103] = sim_ko(sf_los[101], sf_los[102], 103)
        mw[104] = sim_ko(mw[101], mw[102], 104)
        return mw[104]

    print(f"  Monte Carlo ({n_sim:,} simulações)...", flush=True)
    for i in range(n_sim):
        if i % 400 == 0: print(f"    {i/n_sim*100:5.0f}%", end="  ", flush=True)
        standings = {g: sim_group(g, ts) for g, ts in GROUPS.items()}
        champ = run_ko(standings, best_thirds(standings))
        champions[champ] = champions.get(champ, 0) + 1
    print(f"\n  Concluído.")
    return champions


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  CORREÇÕES DE QUALIDADE v11 — Copa 2026")
    print(f"{SEP}\n")

    # ── Carregar raw ──────────────────────────────────────────────────────────
    raw = pd.read_csv(RAW_CSV)
    print(f"  Raw dataset: {len(raw):,} linhas, {raw['match_id'].nunique()} jogos")
    print(f"  DR Congo jogos antes: {raw[raw['team']=='DR Congo']['match_id'].nunique()}")
    print(f"  Congo jogos antes:    {raw[raw['team']=='Congo']['match_id'].nunique()}")

    # ── CORREÇÃO 3 — Levenshtein ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CORREÇÃO 3 — Verificação Levenshtein (possíveis duplicatas ≤ 2)")
    print(f"{SEP}")
    check_levenshtein(raw)

    # ── CORREÇÃO 1 — Normalização de nomes ───────────────────────────────────
    print(f"\n{SEP}")
    print("  CORREÇÃO 1 — Normalização de nomes")
    print(f"{SEP}")
    raw, name_counts = apply_name_corrections(raw)
    print(f"  {'Mapeamento':<35}  {'Linhas corrigidas':>18}")
    print("  " + "-" * 55)
    for old, new in NAME_CORRECTIONS.items():
        cnt = name_counts.get(old, 0)
        status = "✓" if cnt > 0 else "—"
        print(f"  {old:<20} → {new:<14}  {cnt:>14} linhas  {status}")
    print(f"\n  DR Congo jogos após correção: {raw[raw['team']=='DR Congo']['match_id'].nunique()}")

    # ── CORREÇÃO 2 — home_advantage em amistosos ──────────────────────────────
    print(f"\n{SEP}")
    print("  CORREÇÃO 2 — home_advantage=0 para todos os amistosos")
    print(f"{SEP}")
    raw, mh_before, mh_after, ma_before, ma_after, n_changed = apply_friendly_home_advantage(raw)
    print(f"  Amistosos com home_advantage ≠ 0 corrigidos: {n_changed}")
    print(f"  Média gols home — ANTES: {mh_before:.4f}  |  DEPOIS: {mh_after:.4f}")
    print(f"  Média gols away — ANTES: {ma_before:.4f}  |  DEPOIS: {ma_after:.4f}")
    print(f"  Delta home-away — ANTES: {mh_before-ma_before:+.4f}  |  DEPOIS: {mh_after-ma_after:+.4f}")
    note = ("(Atenção: delta positivo persiste — é real, amistosos 'home' são jogados em casa)")
    if abs(mh_after - ma_after) > 0.1:
        print(f"  {note}")

    # ── Salvar raw corrigido ──────────────────────────────────────────────────
    raw.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"\n  Raw salvo: {RAW_CSV}  ({len(raw):,} linhas)")

    # ── Rebuild model_dataset.csv ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Reconstruindo model_dataset.csv...")
    print(f"{SEP}")
    wide = rebuild_model_dataset()
    dr_antes = ((wide["home_team"]=="DR Congo") | (wide["away_team"]=="DR Congo")).sum()
    print(f"  DR Congo em model_dataset: {dr_antes} jogos (antes: ~20 esperado)")

    # ── Saves correction + normalization ──────────────────────────────────────
    wide, global_med, conf_med_game = apply_saves_correction(wide)

    team_conf = {}
    for _, row in raw.iterrows():
        t, c = row["team"], row.get("confederation","")
        if c and c not in ("","FRIENDLY"):
            team_conf[t] = c
    for t, c in COPA_TEAM_CONF.items():
        team_conf.setdefault(t, c)

    global_med_team, conf_medians = compute_conf_medians_by_team(wide, team_conf)
    print(f"  Medianas por confederação:")
    for conf, med in sorted(conf_medians.items()):
        print(f"    {conf:<10}: {med:.4f}")

    long = wide_to_long_v11(wide, team_conf, conf_medians, global_med_team)

    # ── Treino v11 ────────────────────────────────────────────────────────────
    print(f"\n  Treinando xgboost_v11.pkl...")
    model = train_v11(long)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"  Modelo salvo: {OUT_PKL}")

    # ── Lambda cache + MC 2000 ────────────────────────────────────────────────
    team_feats = extract_team_feats(long)
    hf_cohost  = load_cohost_factors()
    lam        = precompute_lambdas(model, team_feats, list(COPA2026_TEAMS))
    champions  = run_mc_2000(lam, hf_cohost, n_sim=2000)
    total      = sum(champions.values())

    # ── Outputs ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DR Congo — comparação n_jogos")
    print(f"{SEP}")
    print(f"  model_dataset — DR Congo jogos: {dr_antes}  (antes correção ~20 | esperado ~22)")

    print(f"\n{SEP}")
    print(f"  Delta gols home vs away em amistosos (verificação modelo)")
    print(f"{SEP}")
    long_fr = long[long["team"].isin(COPA2026_TEAMS)]
    # Amistosos são identificados pelo conf_FRIENDLY ausente (0 em todas conf_*)
    # Na verdade, não temos o tipo de jogo no long_v11. Usar a data e o wide.
    wide_fr = pd.read_csv(MODEL_DATASET, sep=",", parse_dates=["date"])
    if "match_type" in wide_fr.columns:
        fri = wide_fr[wide_fr["match_type"]=="friendly"]
        dh = fri["home_gols"].mean()
        da = fri["away_gols"].mean()
        print(f"  Amistosos (model_dataset) — home: {dh:.4f}  away: {da:.4f}  Δ={dh-da:+.4f}")
    else:
        print("  (match_type não disponível no model_dataset para calcular delta)")

    print(f"\n{SEP}")
    print(f"  Top 15 Campeões (2.000 MC — xgboost_v11 pós-qualidade)")
    print(f"{SEP}")
    sorted_champs = sorted(champions.items(), key=lambda x: x[1], reverse=True)
    for rank, (team, cnt) in enumerate(sorted_champs[:15], 1):
        pct = cnt / total * 100
        bar = "█" * int(pct * 2)
        print(f"  {rank:>2}. {team:<25}  {pct:5.1f}%  {bar}")

    dr_pct = champions.get("DR Congo", 0) / total * 100
    print(f"\n{SEP}")
    print("  Critérios:")
    print(f"  DR Congo < 2.5%  →  {dr_pct:.1f}%   {'✓ OK' if dr_pct < 2.5 else '✗ FALHOU'}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
