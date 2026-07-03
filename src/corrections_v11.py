#!/usr/bin/env python3
"""
corrections_v11.py — Correções cirúrgicas v11

CORREÇÃO 1: Normalizar opp_saves_decay por confederação
    opp_saves_decay_norm = saves_decay / mediana_conf
    Times imputados → 1.0

CORREÇÃO 2: Coletar jogos europeus faltantes via API-Football
    - UEFA Nations League 2022/23: league=5, season=2022
    - UEFA Nations League 2024/25: league=5, season=2024
    - UEFA Euro 2024: league=4, season=2024

Retreinar: xgboost_v11.pkl (15 features)
Monte Carlo: 2.000 simulações
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from xgboost import XGBRegressor

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(encoding="utf-8-sig")

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_CSV         = Path("data/raw/full_dataset_raw.csv")
RAW_BACKUP      = Path("data/raw/full_dataset_raw_pre_v11.csv")
MODEL_DATASET   = Path("data/processed/model_dataset.csv")
MODEL_BACKUP    = Path("data/processed/model_dataset_pre_v11.csv")
HOST_FACTOR_CSV = Path("data/external/host_factor_copa2026.csv")
ELO_CSV         = Path("data/raw/elo_history.csv")
OUT_PKL         = Path("outputs/xgboost_v11.pkl")
CACHE_DIR       = Path("data/raw/api_cache")
CUTOFF          = pd.Timestamp("2026-03-31")
SEED            = 42

# ── Feature constants ─────────────────────────────────────────────────────────
CONF_DUMMIES = [
    "conf_AFC","conf_CAF","conf_CONCACAF","conf_CONMEBOL","conf_OFC","conf_UEFA",
]
RATING_FEATURES = [
    "elo_diff_decay","delta_rating_decay","gls_ponderado_decay",
    "gls_decay_vs_forte","gls_decay_vs_fraco","margem_gols_decay",
]
BASE_FEATURES = ["opp_saves_decay","shots_on_goal_decay","gols_sofridos_decay"]
FEATURES = BASE_FEATURES + RATING_FEATURES + CONF_DUMMIES  # 15 features

XGB_PARAMS = dict(
    objective="count:poisson", n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, random_state=42, n_jobs=-1, verbosity=0,
)

HOST_TEAM_COUNTRY = {
    "United States": "USA",
    "Mexico":        "Mexico",
    "Canada":        "Canada",
}

COPA2026_TEAMS = {
    "Argentina","Brazil","Colombia","Ecuador","Uruguay","Paraguay",
    "Germany","France","Spain","England","Netherlands","Portugal",
    "Belgium","Austria","Switzerland","Croatia",
    "Norway","Sweden","Czech Republic","Turkey",
    "Scotland","Bosnia and Herzegovina",
    "Japan","South Korea","Iran","Australia","Saudi Arabia",
    "Uzbekistan","Jordan","Iraq","Qatar",
    "Morocco","Senegal","Egypt","Ghana","Cape Verde",
    "DR Congo","Ivory Coast","South Africa","Algeria","Tunisia",
    "United States","Mexico","Canada","Panama","Curacao","Haiti",
    "New Zealand",
}

COPA_UEFA_TEAMS = [
    "Germany","France","Spain","England","Netherlands","Portugal",
    "Belgium","Austria","Switzerland","Croatia","Czech Republic",
    "Scotland","Norway","Sweden","Turkey","Bosnia and Herzegovina",
]

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

NAME_MAP = {
    "USA":                    "United States",
    "México":                 "Mexico",
    "Curaçao":                "Curacao",
    "Côte d'Ivoire":          "Ivory Coast",
    "Korea Republic":         "South Korea",
    "Korea DPR":              "North Korea",
    "IR Iran":                "Iran",
    "FYR Macedonia":          "North Macedonia",
    "Bosnia & Herzegovina":   "Bosnia and Herzegovina",
    "Türkiye":                "Turkey",
    "Cape Verde Islands":     "Cape Verde",
    "Congo DR":               "DR Congo",
}

NEW_COMPS = [
    {"name": "UEFA Nations League 2022/23", "confederation": "UEFA",
     "league": 5, "season": 2022, "match_type": "WCQ"},
    {"name": "UEFA Nations League 2024/25", "confederation": "UEFA",
     "league": 5, "season": 2024, "match_type": "WCQ"},
    {"name": "UEFA Euro 2024", "confederation": "UEFA",
     "league": 4, "season": 2024, "match_type": "WCQ"},
]

API_KEY  = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}

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

# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — Coleta API-Football (Correção 2)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(endpoint: str, cache_id: str) -> Path:
    p = CACHE_DIR / endpoint.strip("/") / f"{cache_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def api_get(endpoint: str, params: dict, cache_id: str) -> dict:
    cp = _cache_path(endpoint, cache_id)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            return json.load(f)
    url = f"{BASE_URL}/{endpoint.strip('/')}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30, verify=False)
    resp.raise_for_status()
    data = resp.json()
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    time.sleep(0.5)
    return data


def _parse_val(val):
    if val is None or val == "null" or val == "":
        return None
    if isinstance(val, str):
        s = val.rstrip("%").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return val
    return val


def fetch_competition(comp: dict, existing_ids: set) -> list[dict]:
    league, season = comp["league"], comp["season"]
    data = api_get(
        "fixtures",
        {"league": league, "season": season, "status": "FT"},
        cache_id=f"league{league}_season{season}_FT",
    )
    fixtures = data.get("response", [])
    new_rows: list[dict] = []
    n_skip = n_irrel = 0

    for fixture in fixtures:
        fid      = str(fixture["fixture"]["id"])
        home_raw = fixture["teams"]["home"]["name"]
        away_raw = fixture["teams"]["away"]["name"]
        home_n   = NAME_MAP.get(home_raw, home_raw)
        away_n   = NAME_MAP.get(away_raw, away_raw)

        if home_n not in COPA2026_TEAMS and away_n not in COPA2026_TEAMS:
            n_irrel += 1
            continue
        if fid in existing_ids:
            n_skip += 1
            continue

        goals = fixture["goals"]
        date  = fixture["fixture"]["date"][:10]

        # fetch statistics (cached after first call)
        stats_raw  = api_get("fixtures/statistics", {"fixture": int(fid)}, cache_id=fid)
        team_stats = {}
        for entry in stats_raw.get("response", []):
            tname = entry["team"]["name"]
            team_stats[tname] = {s["type"]: s["value"] for s in entry.get("statistics", [])}

        for side, opp_side, t_raw, o_raw in [
            ("home","away", home_raw, away_raw),
            ("away","home", away_raw, home_raw),
        ]:
            t_n = NAME_MAP.get(t_raw, t_raw)
            o_n = NAME_MAP.get(o_raw, o_raw)
            row: dict = {
                "match_id":        fid,
                "date":            date,
                "competition":     comp["name"],
                "confederation":   comp["confederation"],
                "match_type":      comp["match_type"],
                "season":          season,
                "home_away":       side,
                "team":            t_n,
                "opponent":        o_n,
                "gols_marcados":   goals[side],
                "gols_sofridos":   goals[opp_side],
                "shots":           None,
                "shots_on_goal":   None,
                "blocked_shots":   None,
                "ball_possession": None,
                "fouls":           None,
                "yellow_cards":    None,
                "red_cards":       None,
                "corners":         None,
                "offsides":        None,
                "passes_accurate": None,
                "saves":           None,
                "ast":             None,
                "has_shot_data":   False,
                "weight":          1.0,
                "home_advantage":  1 if side == "home" else 0,
            }
            stats = team_stats.get(t_raw) or team_stats.get(t_n)
            if stats:
                row["has_shot_data"] = True
                for api_type, col in STAT_MAP.items():
                    row[col] = _parse_val(stats.get(api_type))
            new_rows.append(row)

    n_new = len(new_rows) // 2
    print(f"  {comp['name']}: {len(fixtures)} fixtures → "
          f"{n_new} novos | {n_skip} já existiam | {n_irrel} irrelevantes")
    return new_rows


def count_games_per_team(raw_df: pd.DataFrame, teams: list[str]) -> dict[str, int]:
    return {t: raw_df[raw_df["team"] == t]["match_id"].nunique() for t in teams}


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — Reconstrução de model_dataset.csv via build_features
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_model_dataset() -> pd.DataFrame:
    from build_features import (
        load_and_merge_raw, add_decay_features,
        impute_by_confederation, build_match_dataset, XI_DEFAULT, save_data,
    )
    df_raw  = load_and_merge_raw()
    df_dec  = add_decay_features(df_raw, XI_DEFAULT)
    df_imp  = impute_by_confederation(df_dec)
    df_wide = build_match_dataset(df_imp)
    save_data(df_wide)
    return df_wide


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 — Saves correction (zero real vs zero faltante)
# ─────────────────────────────────────────────────────────────────────────────

def apply_saves_correction(wide: pd.DataFrame) -> tuple[pd.DataFrame, float, dict]:
    real_vals: list[float] = []
    for side in ["home","away"]:
        col, imp = f"{side}_saves_decay", f"{side}_saves_decay_imputed"
        if col not in wide.columns:
            continue
        imp_m = wide[imp].astype(bool) if imp in wide.columns else pd.Series(False, index=wide.index)
        real_vals.extend(wide.loc[~imp_m & (wide[col] > 0), col].tolist())
    global_median = float(np.median(real_vals)) if real_vals else 2.5

    conf_med: dict[str, float] = {}
    for conf in CONF_DUMMIES:
        name = conf.replace("conf_","")
        col, imp = "home_saves_decay", "home_saves_decay_imputed"
        if col in wide.columns and conf in wide.columns:
            imp_m = wide[imp].astype(bool) if imp in wide.columns else pd.Series(False, index=wide.index)
            v = wide.loc[wide[conf].astype(bool) & ~imp_m & (wide[col] > 0), col].tolist()
            conf_med[name] = float(np.median(v)) if v else global_median

    fallback = np.full(len(wide), global_median)
    for conf in CONF_DUMMIES:
        name = conf.replace("conf_","")
        if conf in wide.columns:
            fallback = np.where(wide[conf].astype(bool), conf_med.get(name, global_median), fallback)

    wide = wide.copy()
    for side in ["home","away"]:
        saves_col  = f"{side}_saves_decay"
        imp_col    = f"{side}_saves_decay_imputed"
        has_col    = f"{side}_has_decay_data"
        fixed_col  = f"{side}_saves_decay_fixed"
        if saves_col not in wide.columns:
            wide[fixed_col] = global_median
            continue
        fixed    = wide[saves_col].astype(float).copy()
        imp_m    = wide[imp_col].astype(bool) if imp_col in wide.columns else pd.Series(False, index=wide.index)
        has_m    = wide[has_col].astype(bool)  if has_col in wide.columns else pd.Series(True,  index=wide.index)
        zero_mis = (~imp_m) & (fixed == 0) & (~has_m)
        fixed[zero_mis] = pd.Series(fallback, index=wide.index)[zero_mis]
        wide[fixed_col] = fixed

    return wide, global_median, conf_med


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — Normalização por confederação (Correção 1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_conf_medians_by_team(wide: pd.DataFrame,
                                  team_conf: dict) -> tuple[float, dict]:
    """Confederation medians using each team's actual confederation (not game-level conf_)."""
    all_saves: list[pd.DataFrame] = []
    for side in ["home","away"]:
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
        sub.columns = ["team","saves"]
        sub["conf"] = sub["team"].map(team_conf)
        all_saves.append(sub)

    if not all_saves:
        return 2.5, {}
    combined      = pd.concat(all_saves, ignore_index=True)
    conf_medians  = combined.groupby("conf")["saves"].median().to_dict()
    global_median = float(combined["saves"].median())
    return global_median, conf_medians


def wide_to_long_v11(df: pd.DataFrame,
                     team_conf: dict,
                     conf_medians: dict,
                     global_median: float) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in df.iterrows():
        for side, opp in [("home","away"),("away","home")]:
            opp_team = r[f"{opp}_team"]
            own_team = r[f"{side}_team"]

            opp_raw = float(r.get(f"{opp}_saves_decay_fixed", r.get(f"{opp}_saves_decay", 2.5)))
            own_raw = float(r.get(f"{side}_saves_decay_fixed", r.get(f"{side}_saves_decay", 2.5)))

            opp_imp = bool(r.get(f"{opp}_saves_decay_imputed", False))
            own_imp = bool(r.get(f"{side}_saves_decay_imputed", False))

            if opp_imp:
                opp_saves_norm = 1.0
            else:
                opp_c   = team_conf.get(opp_team, "UEFA")
                opp_med = conf_medians.get(opp_c, global_median)
                opp_saves_norm = opp_raw / opp_med if opp_med > 0 else 1.0

            if own_imp:
                own_saves_norm = 1.0
            else:
                own_c   = team_conf.get(own_team, "UEFA")
                own_med = conf_medians.get(own_c, global_median)
                own_saves_norm = own_raw / own_med if own_med > 0 else 1.0

            row: dict = {
                "date":                r["date"],
                "team":                own_team,
                "gols_marcados":       r[f"{side}_gols"],
                "opp_saves_decay":     opp_saves_norm,
                "shots_on_goal_decay": r[f"{side}_shots_on_goal_decay"],
                "gols_sofridos_decay": r.get(f"{side}_gols_sofridos_decay", np.nan),
                "saves_decay":         own_raw,
                "saves_decay_norm":    own_saves_norm,
            }
            for col in CONF_DUMMIES:
                row[col] = int(r.get(col, 0))
            for col in RATING_FEATURES:
                raw = r.get(f"{side}_{col}", np.nan)
                row[col] = float(raw) if pd.notna(raw) else np.nan
            rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 5 — Treinamento v11
# ─────────────────────────────────────────────────────────────────────────────

def train_v11(long: pd.DataFrame) -> XGBRegressor:
    train = long[long["date"] <= CUTOFF]
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURES], train["gols_marcados"])
    return model


def extract_team_feats_v11(long: pd.DataFrame) -> dict:
    team_feats: dict = {}
    for _, row in long.sort_values("date").iterrows():
        entry = {
            # saves_decay_norm armazenado sob 'saves_decay' para ser usado como
            # opp_saves_decay em precompute_lambdas (f2.get("saves_decay", ...))
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
        for team, val in (elo.sort_values("date")
                           .groupby("team")["elo_after"].last().items()):
            team_feats.setdefault(team, {})["_elo_current"] = float(val)
    return team_feats


def precompute_lambdas_v11(model: XGBRegressor,
                            team_feats: dict,
                            teams: list[str]) -> dict:
    rows, pairs = [], []
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            f1, f2 = team_feats.get(t1, {}), team_feats.get(t2, {})
            row: dict = {
                "opp_saves_decay":     f2.get("saves_decay", 1.0),  # normalized
                "shots_on_goal_decay": f1.get("shots_on_goal_decay", 3.5),
                "gols_sofridos_decay": f1.get("gols_sofridos_decay", np.nan),
                **{c: 0 for c in CONF_DUMMIES},
            }
            for col in RATING_FEATURES:
                if col == "elo_diff_decay":
                    e1, e2 = f1.get("_elo_current", np.nan), f2.get("_elo_current", np.nan)
                    row[col] = (float(e1) - float(e2)
                                if pd.notna(e1) and pd.notna(e2)
                                else f1.get(col, np.nan))
                else:
                    row[col] = f1.get(col, np.nan)
            rows.append(row)
            pairs.append((t1, t2))
    preds = model.predict(pd.DataFrame(rows)[FEATURES])
    return {pair: float(p) for pair, p in zip(pairs, preds)}


def load_cohost_factors() -> dict[str, float]:
    df = pd.read_csv(HOST_FACTOR_CSV)
    return {row["team"]: float(row["host_factor_cohost"]) for _, row in df.iterrows()}


def get_match_lambdas(t1: str, t2: str, venue: str,
                      lam: dict, hf: dict) -> tuple[float, float]:
    def _bonus(team: str, v: str) -> float:
        tc = HOST_TEAM_COUNTRY.get(team)
        return hf.get(team, 0.0) * 0.5 if (tc and tc == v) else 0.0
    lh = lam.get((t1, t2), 1.2) * (1 + _bonus(t1, venue))
    la = lam.get((t2, t1), 1.0) * (1 + _bonus(t2, venue))
    return lh, la


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 6 — Monte Carlo 2.000 simulações
# ─────────────────────────────────────────────────────────────────────────────

def run_mc_2000(lam: dict, hf: dict, n_sim: int = 2000) -> dict[str, int]:
    from simulate_copa2026 import (
        GROUPS, R32, MATCH_VENUES, GROUP_VENUE_COUNTRY,
        assign_thirds, simulate_game_from_lambdas,
    )

    random.seed(SEED)
    np.random.seed(SEED)
    champions: dict[str, int] = {}

    def sim_group(gid: str, teams: list, stochastic: bool) -> pd.DataFrame:
        venue = GROUP_VENUE_COUNTRY[gid]
        stats = {t: {"pts":0,"gf":0,"ga":0,"gd":0} for t in teams}
        for t1, t2 in combinations(teams, 2):
            lh, la = get_match_lambdas(t1, t2, venue, lam, hf)
            g1, g2 = simulate_game_from_lambdas(lh, la, stochastic)
            if   g1 > g2: stats[t1]["pts"] += 3
            elif g2 > g1: stats[t2]["pts"] += 3
            else: stats[t1]["pts"] += 1; stats[t2]["pts"] += 1
            for t, gf, ga in [(t1,g1,g2),(t2,g2,g1)]:
                stats[t]["gf"] += gf; stats[t]["ga"] += ga; stats[t]["gd"] += gf - ga
        df = pd.DataFrame([{"team":t, **v} for t, v in stats.items()])
        df = df.sort_values(["pts","gd","gf"], ascending=False).reset_index(drop=True)
        df["pos"] = df.index + 1
        return df

    def best_thirds(standings: dict) -> list[dict]:
        thirds = []
        for g, df in standings.items():
            row = df[df["pos"] == 3].iloc[0]
            thirds.append({
                "group": g, "team": row["team"],
                "pts": row["pts"], "gd": row["gd"],
                "gf": row["gf"], "_tb": random.random(),
            })
        thirds.sort(key=lambda x: (x["pts"],x["gd"],x["gf"],x["_tb"]), reverse=True)
        return thirds[:8]

    def sim_ko(t1: str, t2: str, mid: int) -> str:
        venue = MATCH_VENUES.get(mid, "USA")
        lh, la = get_match_lambdas(t1, t2, venue, lam, hf)
        g1, g2 = int(np.random.poisson(lh)), int(np.random.poisson(la))
        if g1 != g2:
            return t1 if g1 > g2 else t2
        g1 += int(np.random.poisson(lh / 3))
        g2 += int(np.random.poisson(la / 3))
        if g1 != g2:
            return t1 if g1 > g2 else t2
        return t1 if random.random() < 0.5 else t2

    def run_ko(standings: dict, q_thirds: list) -> str:
        t_by_grp = {t["group"]: t["team"] for t in q_thirds}
        t_asgn   = assign_thirds(q_thirds)

        def pos(g: str, n: int) -> str:
            return standings[g][standings[g]["pos"] == n].iloc[0]["team"]

        def resolve(slot: str, mid: int) -> str:
            if slot[0] == "1": return pos(slot[1], 1)
            if slot[0] == "2": return pos(slot[1], 2)
            g = t_asgn.get(mid)
            return t_by_grp.get(g, next(iter(t_by_grp.values())))

        mw: dict[int, str] = {}
        for mid, a_slot, b_slot, _ in R32:
            mw[mid] = sim_ko(resolve(a_slot, mid), resolve(b_slot, mid), mid)

        for mid, m1, m2 in [(89,73,74),(90,75,76),(91,77,78),(92,79,80),
                             (93,81,82),(94,83,84),(95,85,86),(96,87,88)]:
            mw[mid] = sim_ko(mw[m1], mw[m2], mid)

        for mid, m1, m2 in [(97,89,90),(98,91,92),(99,93,94),(100,95,96)]:
            mw[mid] = sim_ko(mw[m1], mw[m2], mid)

        sf_los: dict[int, str] = {}
        for mid, m1, m2 in [(101,97,98),(102,99,100)]:
            t1, t2 = mw[m1], mw[m2]
            w = sim_ko(t1, t2, mid)
            mw[mid] = w
            sf_los[mid] = t2 if w == t1 else t1

        mw[103] = sim_ko(sf_los[101], sf_los[102], 103)
        mw[104] = sim_ko(mw[101], mw[102], 104)
        return mw[104]

    print(f"  Monte Carlo ({n_sim:,} simulações)...", flush=True)
    for i in range(n_sim):
        if i % 400 == 0:
            print(f"    {i/n_sim*100:5.0f}%", end="  ", flush=True)
        standings = {g: sim_group(g, ts, stochastic=True) for g, ts in GROUPS.items()}
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
    print("  CORREÇÕES CIRÚRGICAS v11 — Copa 2026")
    print(f"  Correção 1: normalizar opp_saves_decay por confederação")
    print(f"  Correção 2: adicionar UEFA NL + Euro 2024 ao dataset")
    print(f"{SEP}\n")

    # ── 0. Backup dos arquivos originais ──────────────────────────────────────
    if not RAW_BACKUP.exists():
        copyfile(RAW_CSV, RAW_BACKUP)
        print(f"  Backup: {RAW_BACKUP}")
    else:
        print(f"  Backup já existe: {RAW_BACKUP}")

    if MODEL_DATASET.exists() and not MODEL_BACKUP.exists():
        copyfile(MODEL_DATASET, MODEL_BACKUP)
        print(f"  Backup: {MODEL_BACKUP}")

    # ── 1. N_jogos antes (base = pré-v11) ─────────────────────────────────────
    raw_orig = pd.read_csv(RAW_BACKUP)
    n_antes = count_games_per_team(raw_orig, COPA_UEFA_TEAMS)

    # ── 2. Correção 2 — Fetch jogos europeus faltantes ────────────────────────
    print(f"\n{SEP}")
    print("  CORREÇÃO 2 — Coletando jogos europeus via API-Football")
    print(f"{SEP}")

    existing_ids = set(raw_orig["match_id"].astype(str).unique())
    print(f"  Match IDs existentes: {len(existing_ids)}")

    all_new_rows: list[dict] = []
    for comp in NEW_COMPS:
        rows = fetch_competition(comp, existing_ids)
        all_new_rows.extend(rows)

    print(f"\n  Total de novas linhas: {len(all_new_rows)} "
          f"({len(all_new_rows)//2} jogos)")

    if all_new_rows:
        new_df    = pd.DataFrame(all_new_rows)
        augmented = pd.concat([raw_orig, new_df], ignore_index=True)
        augmented.to_csv(RAW_CSV, index=False, encoding="utf-8")
        print(f"  Salvo: {RAW_CSV}  ({len(augmented):,} linhas)")
    else:
        augmented = raw_orig
        print("  Nenhum jogo novo encontrado.")

    # ── 3. N_jogos depois ─────────────────────────────────────────────────────
    n_depois = count_games_per_team(augmented, COPA_UEFA_TEAMS)

    # ── 4. Reconstruir model_dataset.csv ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  Reconstruindo model_dataset.csv via build_features...")
    print(f"{SEP}")
    wide = rebuild_model_dataset()
    print(f"  Dataset wide: {len(wide)} jogos × {len(wide.columns)} colunas")

    # ── 5. Correção 3 — zeros faltantes em saves_decay ────────────────────────
    wide, global_med_game, conf_med_game = apply_saves_correction(wide)

    # ── 6. Correção 1 — normalizar opp_saves_decay por confederação ───────────
    print(f"\n{SEP}")
    print("  CORREÇÃO 1 — Normalizando opp_saves_decay por confederação")
    print(f"{SEP}")

    # Build team→confederation map from augmented raw dataset
    team_conf = {}
    for _, row in augmented.iterrows():
        t = row["team"]
        c = row.get("confederation", "")
        if c and c not in ("", "FRIENDLY"):
            team_conf[t] = c
    # Preencher copa teams sem entrada (caso inexistam no raw)
    for t, c in COPA_TEAM_CONF.items():
        if t not in team_conf:
            team_conf[t] = c

    global_med, conf_medians = compute_conf_medians_by_team(wide, team_conf)

    print(f"  Mediana global de saves_decay: {global_med:.4f}")
    print(f"  Medianas por confederação (usadas para normalização):")
    for conf, med in sorted(conf_medians.items()):
        print(f"    {conf:<10}: {med:.4f}")

    # ── 7. Wide → Long com normalização ───────────────────────────────────────
    print(f"\n  Construindo dataset long com opp_saves normalizado...")
    long = wide_to_long_v11(wide, team_conf, conf_medians, global_med)
    print(f"  Dataset long: {len(long)} linhas")

    # ── 8. Treinar xgboost_v11.pkl ────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  Treinando xgboost_v11.pkl")
    print(f"  Features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"{SEP}")
    model_v11 = train_v11(long)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(model_v11, f)
    print(f"  Modelo salvo: {OUT_PKL}")

    # ── 9. Extrair team_feats e lambdas ───────────────────────────────────────
    team_feats = extract_team_feats_v11(long)
    hf_cohost  = load_cohost_factors()
    all_teams  = list(COPA2026_TEAMS)
    lam        = precompute_lambdas_v11(model_v11, team_feats, all_teams)
    print(f"  Lambdas pré-computados: {len(lam)} pares")

    # ── 10. Monte Carlo 2.000 ─────────────────────────────────────────────────
    champions = run_mc_2000(lam, hf_cohost, n_sim=2000)
    total = sum(champions.values())

    # ─── OUTPUTS ─────────────────────────────────────────────────────────────

    print(f"\n{'=' * 72}")
    print("  OUTPUT 1 — n_jogos antes/depois por time europeu Copa 2026")
    print(f"{'=' * 72}")
    print(f"  {'Time':<25}  {'Antes':>7}  {'Depois':>7}  {'Δ':>5}")
    print("  " + "-" * 48)
    for t in sorted(COPA_UEFA_TEAMS):
        a = n_antes.get(t, 0)
        d = n_depois.get(t, 0)
        print(f"  {t:<25}  {a:>7}  {d:>7}  {d-a:>+5}")

    print(f"\n{'=' * 72}")
    print("  OUTPUT 2 — opp_saves_decay: original vs normalizado (10 mais extremos)")
    print(f"{'=' * 72}")
    print(f"  {'Time':<25}  {'Conf':<10}  {'saves_raw':>10}  {'conf_med':>10}  {'saves_norm':>11}  {'|norm-1|':>9}")
    print("  " + "-" * 78)

    saves_comparison = []
    for team, feats in team_feats.items():
        if team not in COPA2026_TEAMS:
            continue
        raw    = feats.get("saves_decay_raw", np.nan)
        norm   = feats.get("saves_decay", np.nan)  # normalized stored under saves_decay
        conf   = COPA_TEAM_CONF.get(team, team_conf.get(team, "?"))
        med    = conf_medians.get(conf, global_med)
        if pd.notna(raw) and pd.notna(norm):
            saves_comparison.append({
                "team": team, "conf": conf,
                "saves_raw": raw, "conf_med": med, "saves_norm": norm,
                "abs_diff": abs(norm - 1.0),
            })

    saves_comparison.sort(key=lambda x: x["abs_diff"], reverse=True)
    for row in saves_comparison[:10]:
        print(f"  {row['team']:<25}  {row['conf']:<10}  "
              f"{row['saves_raw']:>10.4f}  {row['conf_med']:>10.4f}  "
              f"{row['saves_norm']:>11.4f}  {row['abs_diff']:>9.4f}")

    print(f"\n{'=' * 72}")
    print(f"  OUTPUT 3 — Top 15 Campeões (2.000 MC — xgboost_v11)")
    print(f"{'=' * 72}")
    sorted_champs = sorted(champions.items(), key=lambda x: x[1], reverse=True)
    for rank, (team, cnt) in enumerate(sorted_champs[:15], 1):
        pct = cnt / total * 100
        bar = "█" * int(pct * 2)
        print(f"  {rank:>2}. {team:<25}  {pct:5.1f}%  {bar}")

    print(f"\n{'=' * 72}")
    print("  Critérios de aceite:")
    accept = {
        "França > 5%":     ("France",   champions.get("France",   0)/total*100,  5.0, ">"),
        "Japão < 4%":      ("Japan",    champions.get("Japan",    0)/total*100,  4.0, "<"),
        "Brasil > 3%":     ("Brazil",   champions.get("Brazil",   0)/total*100,  3.0, ">"),
        "Portugal > 3%":   ("Portugal", champions.get("Portugal", 0)/total*100,  3.0, ">"),
    }
    all_ok = True
    for label, (team, pct, threshold, op) in accept.items():
        ok = (pct > threshold) if op == ">" else (pct < threshold)
        status = "✓ OK" if ok else "✗ FALHOU"
        if not ok:
            all_ok = False
        print(f"  {label:<20} {team}: {pct:.1f}%   {status}")
    print(f"\n  {'✓ TODOS OS CRITÉRIOS ATENDIDOS' if all_ok else '✗ ALGUNS CRITÉRIOS NÃO ATENDIDOS'}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
