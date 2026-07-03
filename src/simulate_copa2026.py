"""
simulate_copa2026.py — Simulação completa da Copa 2026 com XGBoost count:poisson

Modelo v10:
  - Features v10: 16 features (mesmo conjunto do v9, opp_saves_decay restaurado)
  - host_factor venue-aware: usa host_factor_home/cohost/zero por jogo
  - HOST_TEAM_COUNTRY + GROUP_VENUE_COUNTRY para mapeamento completo
  - get_host_factor(team, venue_country) → 0|0.5×|1×

Saídas:
  outputs/xgboost_v10.pkl
  outputs/copa2026_previsoes_v7.csv
  outputs/predicoes_rodada1_v10.csv
"""

import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from scipy.stats import poisson as scipy_poisson
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INPUT_CSV        = Path("data/processed/model_dataset.csv")
HOST_FACTOR_CSV  = Path("data/external/host_factor_copa2026.csv")
OUT_TXT          = Path("outputs/copa2026_bracket.txt")
OUT_MC           = Path("outputs/copa2026_previsoes_v7.csv")
OUT_MC_PREV      = Path("outputs/copa2026_previsoes_v6.csv")
OUT_PKL          = Path("outputs/xgboost_v10.pkl")
OUT_R1           = Path("outputs/predicoes_rodada1_v10.csv")
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

N_SIM  = 10_000
SEED   = 42
CUTOFF = "2026-03-31"

XGB_PARAMS = dict(
    objective="count:poisson", n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, random_state=42, n_jobs=-1, verbosity=0,
)

# ---------------------------------------------------------------------------
# Features v10 — mesmo conjunto do v9 (opp_saves_decay restaurado após MAE +0.0244
# exceder threshold e Panama inflacionar para 6.7% sem a penalização de goleira)
# ---------------------------------------------------------------------------
BASE_FEATURES = [
    "opp_saves_decay",
    "shots_on_goal_decay",
    "gols_sofridos_decay",
    "host_factor",
]
RATING_FEATURES = [
    "elo_diff_decay",
    "delta_rating_decay",
    "gls_ponderado_decay",
    "gls_decay_vs_forte",
    "gls_decay_vs_fraco",
    "margem_gols_decay",
]
CONF_DUMMIES = [
    "conf_AFC", "conf_CAF", "conf_CONCACAF", "conf_CONMEBOL",
    "conf_OFC", "conf_UEFA",
]
FEATURES = BASE_FEATURES + RATING_FEATURES + CONF_DUMMIES  # 16 features

# ---------------------------------------------------------------------------
# Grupos — chaveamento oficial Copa 2026
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",        "South Korea",  "South Africa",        "Czech Republic"],
    "B": ["Canada",        "Switzerland",  "Qatar",               "Bosnia and Herzegovina"],
    "C": ["Brazil",        "Morocco",      "Haiti",               "Scotland"],
    "D": ["United States", "Paraguay",     "Australia",           "Turkey"],
    "E": ["Germany",       "Ivory Coast",  "Ecuador",             "Curacao"],
    "F": ["Netherlands",   "Sweden",       "Tunisia",             "Japan"],
    "G": ["Belgium",       "Iran",         "New Zealand",         "Egypt"],
    "H": ["Spain",         "Saudi Arabia", "Uruguay",             "Cape Verde"],
    "I": ["France",        "Senegal",      "Iraq",                "Norway"],
    "J": ["Argentina",     "Algeria",      "Austria",             "Jordan"],
    "K": ["Portugal",      "DR Congo",     "Uzbekistan",          "Colombia"],
    "L": ["England",       "Croatia",      "Ghana",               "Panama"],
}

TEAM_TO_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}
HOST_NATIONS  = {"United States", "Mexico", "Canada"}

# País de origem de cada time-sede (team → venue_country_code)
HOST_TEAM_COUNTRY: dict[str, str] = {
    "United States": "USA",
    "Mexico":        "Mexico",
    "Canada":        "Canada",
}

# País que sedia cada grupo (baseado no calendário oficial FIFA Copa 2026)
# Grupo A: Mexico City / GDL / MTY → Mexico
# Grupo B: Toronto / Vancouver → Canada  (Seattle é sede auxiliar: simplificamos para Canada)
# Grupo D: Dallas / Los Angeles / San Francisco → USA
# Todos os outros grupos: USA
GROUP_VENUE_COUNTRY: dict[str, str] = {
    "A": "Mexico",
    "B": "Canada",
    "C": "USA", "D": "USA", "E": "USA", "F": "USA",
    "G": "USA", "H": "USA", "I": "USA", "J": "USA",
    "K": "USA", "L": "USA",
}

# Sede do mata-mata (calendário oficial FIFA, dez/2024)
MATCH_VENUES: dict[int, str] = {
    # R32 (dezesseis-avos) M73-M88
    73: "USA",    # MetLife NJ — 2A vs 2B
    74: "USA",    # AT&T Dallas — 1E vs 3rd
    75: "USA",    # Levi's SF — 1F vs 2C
    76: "USA",    # Rose Bowl — 1C vs 2F
    77: "USA",    # SoFi LA — 1I vs 3rd
    78: "USA",    # Arrowhead KC — 2E vs 2I
    79: "USA",    # Levi's SF — 1A vs 3rd  ← Mexico (1A) fora de casa!
    80: "USA",    # MetLife — 1L vs 3rd
    81: "USA",    # Lincoln Financial PHI — 1D vs 3rd
    82: "USA",    # Hard Rock Miami — 1G vs 3rd
    83: "USA",    # NRG Houston — 2K vs 2L
    84: "USA",    # Mercedes-Benz Atlanta — 1H vs 2J
    85: "Mexico", # Estadio Azteca MX — 1B vs 3rd  ← Canada (1B) fora de casa!
    86: "USA",    # AT&T Dallas — 1J vs 2H
    87: "Mexico", # Estadio Akron GDL — 1K vs 3rd
    88: "Mexico", # Estadio BBVA MTY — 2D vs 2G  ← USA (2D) fora de casa!
    # R16 (oitavas) M89-M96
    89: "USA",  90: "USA",  91: "USA",  92: "USA",
    93: "USA",  94: "USA",  95: "Mexico",  96: "Mexico",
    # QF (quartas) M97-M100
    97: "USA",  98: "USA",  99: "USA",  100: "Mexico",
    # SF M101-M102
    101: "USA", 102: "USA",
    # 3º + Final
    103: "USA", 104: "USA",
}

# Jogos da 1ª Rodada (11-13/jun/2026)
# (team1, team2, date_str, group, venue_country)
ROUND1_GAMES: list[tuple] = [
    ("Mexico",         "South Africa",          "2026-06-11", "A", "Mexico"),
    ("South Korea",    "Czech Republic",        "2026-06-11", "A", "Mexico"),
    ("Canada",         "Bosnia and Herzegovina","2026-06-12", "B", "Canada"),
    ("United States",  "Paraguay",              "2026-06-12", "D", "USA"),
    ("Spain",          "Cape Verde",            "2026-06-12", "H", "USA"),
    ("Brazil",         "Morocco",               "2026-06-13", "C", "USA"),
    ("Austria",        "Jordan",                "2026-06-13", "J", "USA"),
    ("Turkey",         "Australia",             "2026-06-13", "D", "USA"),
]

R32: list[tuple] = [
    (73,  "2A",  "2B",  None),
    (74,  "1E",  "3",   ["A","B","C","D","F"]),
    (75,  "1F",  "2C",  None),
    (76,  "1C",  "2F",  None),
    (77,  "1I",  "3",   ["C","D","F","G","H"]),
    (78,  "2E",  "2I",  None),
    (79,  "1A",  "3",   ["C","E","F","H","I"]),
    (80,  "1L",  "3",   ["E","H","I","J","K"]),
    (81,  "1D",  "3",   ["B","E","F","I","J"]),
    (82,  "1G",  "3",   ["A","E","H","I","J"]),
    (83,  "2K",  "2L",  None),
    (84,  "1H",  "2J",  None),
    (85,  "1B",  "3",   ["E","F","G","I","J"]),
    (86,  "1J",  "2H",  None),
    (87,  "1K",  "3",   ["D","E","I","J","L"]),
    (88,  "2D",  "2G",  None),
]

STAGE_KEY = {
    "grupo":     "Fase de Grupos",
    "r32":       "Oitavas (R32)",
    "r16":       "Quartas (R16)",
    "qf":        "Semifinal (QF)",
    "sf":        "Final (SF)",
    "3rd_place": "3º Lugar",
    "champion":  "Campeão",
}


# ---------------------------------------------------------------------------
# Host factor helpers
# ---------------------------------------------------------------------------

def load_host_factors() -> tuple[dict[str, float], dict[str, float]]:
    """Carrega host_factor_home e host_factor_cohost do CSV externo."""
    if not HOST_FACTOR_CSV.exists():
        print(f"  AVISO: {HOST_FACTOR_CSV} não encontrado — host_factor = 0 para todos")
        return {}, {}
    hf_df = pd.read_csv(HOST_FACTOR_CSV)
    hf_home   = {r["team"]: float(r["host_factor_home"])   for _, r in hf_df.iterrows()}
    hf_cohost = {r["team"]: float(r["host_factor_cohost"]) for _, r in hf_df.iterrows()}
    return hf_home, hf_cohost


def get_host_factor(team: str, venue_country: str,
                    hf_home: dict[str, float],
                    hf_cohost: dict[str, float]) -> float:
    """Retorna host_factor correto para um time em uma sede específica.

    - team é sede E joga no próprio país  → host_factor_home
    - team é sede E joga em outro co-sede → host_factor_cohost (≈ 0.5×home)
    - caso contrário                       → 0.0
    """
    team_country = HOST_TEAM_COUNTRY.get(team)
    if team_country is None:
        return 0.0
    if team_country == venue_country:
        return hf_home.get(team, 0.0)
    if venue_country in HOST_TEAM_COUNTRY.values():
        return hf_cohost.get(team, 0.0)
    return 0.0


def _get_team_cache(team: str, venue_country: str,
                    lam_full, lam_cohost, lam_neutral):
    """Seleciona o lambda cache correto para um time em uma sede."""
    tc = HOST_TEAM_COUNTRY.get(team)
    if tc is None:
        return lam_neutral
    if tc == venue_country:
        return lam_full
    if venue_country in HOST_TEAM_COUNTRY.values():
        return lam_cohost
    return lam_neutral


def get_match_lambdas(t1: str, t2: str, venue_country: str,
                      lam_full, lam_cohost, lam_neutral) -> tuple[float, float]:
    """Retorna (lh, la) com host_factor correto para a sede do jogo."""
    lc1 = _get_team_cache(t1, venue_country, lam_full, lam_cohost, lam_neutral)
    lc2 = _get_team_cache(t2, venue_country, lam_full, lam_cohost, lam_neutral)
    return lc1.get((t1, t2), 1.2), lc2.get((t2, t1), 1.0)


# ---------------------------------------------------------------------------
# Validação: imprimir host_factor aplicado por jogo
# ---------------------------------------------------------------------------

def validate_host_factors(hf_home: dict, hf_cohost: dict) -> None:
    sep = "=" * 75
    print(f"\n{sep}")
    print("  VALIDAÇÃO host_factor — Copa 2026 v10 (venue-aware)")
    print(sep)
    print(f"  {'Jogo':35s}  {'Venue':8s}  {'HF_Time1':>9}  {'HF_Time2':>9}  {'OK?':>5}")
    print("  " + "-" * 72)

    test_cases = [
        ("Mexico",        "South Africa",         "Mexico",  ">0",  "=0"),
        ("United States", "Paraguay",             "USA",     ">0",  "=0"),
        ("Mexico",        "Canada",               "USA",     "cohost","cohost"),
        ("Brazil",        "Morocco",              "USA",     "=0",  "=0"),
        ("Argentina",     "Austria",              "USA",     "=0",  "=0"),
        ("Mexico",        "South Korea",          "Mexico",  ">0",  "=0"),
        ("Canada",        "Bosnia and Herzegovina","Canada", ">0",  "=0"),
        ("Mexico",        "Czech Republic",       "USA",     "cohost","=0"),
    ]
    for t1, t2, venue, exp1, exp2 in test_cases:
        hf1 = get_host_factor(t1, venue, hf_home, hf_cohost)
        hf2 = get_host_factor(t2, venue, hf_home, hf_cohost)
        ok1 = (hf1 > 0 if exp1 == ">0" else
               (abs(hf1 - hf_cohost.get(t1, 0)) < 0.001 if exp1 == "cohost" else
                abs(hf1) < 0.001))
        ok2 = (hf2 > 0 if exp2 == ">0" else
               (abs(hf2 - hf_cohost.get(t2, 0)) < 0.001 if exp2 == "cohost" else
                abs(hf2) < 0.001))
        status = "✓" if ok1 and ok2 else "✗ BUG"
        print(f"  {t1+' × '+t2:35s}  {venue:8s}  {hf1:9.4f}  {hf2:9.4f}  {status:>5}")
    print(sep)


# ---------------------------------------------------------------------------
# Carregar + treinar
# ---------------------------------------------------------------------------

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for side, opp in [("home", "away"), ("away", "home")]:
            hf = int(r.get("home_advantage", 0)) if side == "home" else 0
            row = {
                "match_id":            r["match_id"],
                "date":                r["date"],
                "team":                r[f"{side}_team"],
                "side":                side,
                "gols_marcados":       r[f"{side}_gols"],
                "shots_on_goal_decay": r[f"{side}_shots_on_goal_decay"],
                "gols_sofridos_decay": r.get(f"{side}_gols_sofridos_decay", np.nan),
                "opp_saves_decay":     float(r.get(f"{opp}_saves_decay", 2.0)),
                "saves_decay":         float(r.get(f"{side}_saves_decay", 2.0)),
                "host_factor":         hf,
            }
            for col in CONF_DUMMIES:
                row[col] = int(r.get(col, 0))
            for col in RATING_FEATURES:
                raw = r.get(f"{side}_{col}", np.nan)
                row[col] = float(raw) if pd.notna(raw) else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def train_model() -> tuple[XGBRegressor, dict[str, dict]]:
    wide = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    core = [c for c in wide.columns if c.endswith("_decay") and not c.startswith("has_")]
    wide = wide.dropna(subset=core).reset_index(drop=True)
    long = wide_to_long(wide)

    train = long[long["date"] <= CUTOFF]
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURES], train["gols_marcados"])

    team_feats: dict[str, dict] = {}
    for _, row in long.sort_values("date").iterrows():
        entry: dict = {
            "shots_on_goal_decay": row.get("shots_on_goal_decay", 3.5),
            "gols_sofridos_decay": row.get("gols_sofridos_decay", np.nan),
            "saves_decay":         row.get("saves_decay", 2.0),
        }
        for col in RATING_FEATURES:
            v = row.get(col, np.nan)
            entry[col] = float(v) if pd.notna(v) else np.nan
        team_feats[row["team"]] = entry

    # ELO corrente
    elo_path = Path("data/raw/elo_history.csv")
    if elo_path.exists():
        elo_hist = pd.read_csv(elo_path, parse_dates=["date"])
        current_elo = (elo_hist.sort_values("date")
                       .groupby("team")["elo_after"].last()
                       .to_dict())
        for team, elo_val in current_elo.items():
            if team in team_feats:
                team_feats[team]["_elo_current"] = float(elo_val)
            else:
                team_feats[team] = {"_elo_current": float(elo_val)}

    with open(OUT_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"  Modelo salvo: {OUT_PKL}")
    return model, team_feats


# ---------------------------------------------------------------------------
# Pre-computação de lambdas (venue-aware)
# ---------------------------------------------------------------------------

def precompute_lambdas(model: XGBRegressor,
                       team_feats: dict,
                       teams: list[str],
                       hf_dict: dict[str, float]) -> dict[tuple[str, str], float]:
    """Computa lambdas com host_factor dado por hf_dict.

    Passar {} para lam_neutral, hf_home para lam_full, hf_cohost para lam_cohost.
    """
    rows, pairs = [], []
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            f1 = team_feats.get(t1, {})
            f2 = team_feats.get(t2, {})
            row = {
                "opp_saves_decay":     f2.get("saves_decay", 2.0),
                "shots_on_goal_decay": f1.get("shots_on_goal_decay", 3.5),
                "gols_sofridos_decay": f1.get("gols_sofridos_decay", np.nan),
                "host_factor":         hf_dict.get(t1, 0.0),
                "conf_AFC": 0, "conf_CAF": 0, "conf_CONCACAF": 0,
                "conf_CONMEBOL": 0, "conf_OFC": 0, "conf_UEFA": 0,
            }
            for col in RATING_FEATURES:
                if col == "elo_diff_decay":
                    elo1 = f1.get("_elo_current", np.nan)
                    elo2 = f2.get("_elo_current", np.nan)
                    row[col] = (float(elo1) - float(elo2)
                                if pd.notna(elo1) and pd.notna(elo2)
                                else f1.get(col, np.nan))
                else:
                    row[col] = f1.get(col, np.nan)
            rows.append(row)
            pairs.append((t1, t2))

    df    = pd.DataFrame(rows)
    preds = model.predict(df[FEATURES])
    return {pair: float(lam) for pair, lam in zip(pairs, preds)}


# ---------------------------------------------------------------------------
# Simulação de jogo
# ---------------------------------------------------------------------------

def simulate_game_from_lambdas(lh: float, la: float,
                                stochastic: bool) -> tuple[int, int]:
    if stochastic:
        return int(np.random.poisson(lh)), int(np.random.poisson(la))
    ph, pa, best = 0, 0, -1.0
    for h in range(9):
        for a in range(9):
            p = scipy_poisson.pmf(h, lh) * scipy_poisson.pmf(a, la)
            if p > best:
                best, ph, pa = p, h, a
    return ph, pa


def simulate_knockout_game(t1: str, t2: str,
                            match_id: int,
                            lam_full, lam_cohost, lam_neutral,
                            stochastic: bool = True) -> tuple[str, str, str]:
    venue_country = MATCH_VENUES.get(match_id, "USA")
    lh, la = get_match_lambdas(t1, t2, venue_country, lam_full, lam_cohost, lam_neutral)

    if stochastic:
        g1, g2 = int(np.random.poisson(lh)), int(np.random.poisson(la))
    else:
        g1, g2 = simulate_game_from_lambdas(lh, la, stochastic=False)

    if g1 != g2:
        return (t1, t2, "normal") if g1 > g2 else (t2, t1, "normal")

    e1 = int(np.random.poisson(lh / 3)) if stochastic else 0
    e2 = int(np.random.poisson(la / 3)) if stochastic else 0
    g1 += e1; g2 += e2

    if g1 != g2:
        return (t1, t2, "prorrogacao") if g1 > g2 else (t2, t1, "prorrogacao")

    winner = (t1 if np.random.random() < 0.5 else t2) if stochastic else (t1 if lh >= la else t2)
    loser  = t2 if winner == t1 else t1
    return winner, loser, "penaltis"


# ---------------------------------------------------------------------------
# Fase de Grupos
# ---------------------------------------------------------------------------

def simulate_group(group_id: str, group_teams: list[str],
                   lam_full, lam_cohost, lam_neutral,
                   stochastic: bool) -> pd.DataFrame:
    venue_country = GROUP_VENUE_COUNTRY[group_id]
    stats = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in group_teams}
    for t1, t2 in combinations(group_teams, 2):
        lh, la = get_match_lambdas(t1, t2, venue_country, lam_full, lam_cohost, lam_neutral)
        g1, g2 = simulate_game_from_lambdas(lh, la, stochastic)
        if g1 > g2:   stats[t1]["pts"] += 3
        elif g2 > g1: stats[t2]["pts"] += 3
        else:         stats[t1]["pts"] += 1; stats[t2]["pts"] += 1
        for t, gf, ga in [(t1, g1, g2), (t2, g2, g1)]:
            stats[t]["gf"] += gf; stats[t]["ga"] += ga; stats[t]["gd"] += gf - ga

    df = pd.DataFrame([{"team": t, **v} for t, v in stats.items()])
    df = df.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
    df["pos"] = df.index + 1
    return df


def simulate_all_groups(lam_full, lam_cohost, lam_neutral,
                        stochastic: bool) -> dict[str, pd.DataFrame]:
    return {g: simulate_group(g, ts, lam_full, lam_cohost, lam_neutral, stochastic)
            for g, ts in GROUPS.items()}


# ---------------------------------------------------------------------------
# 8 melhores 3ºs
# ---------------------------------------------------------------------------

def best_third_place(standings: dict[str, pd.DataFrame]) -> list[dict]:
    thirds = []
    for g, df in standings.items():
        row = df[df["pos"] == 3].iloc[0]
        thirds.append({"group": g, "team": row["team"],
                        "pts": row["pts"], "gd": row["gd"], "gf": row["gf"]})
    thirds.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    return thirds[:8]


# ---------------------------------------------------------------------------
# Atribuição dos 3ºs (matching bipartido)
# ---------------------------------------------------------------------------

THIRD_ELIGIBLE: dict[int, list[str]] = {
    74: ["A","B","C","D","F"],  77: ["C","D","F","G","H"],
    79: ["C","E","F","H","I"],  80: ["E","H","I","J","K"],
    81: ["B","E","F","I","J"],  82: ["A","E","H","I","J"],
    85: ["E","F","G","I","J"],  87: ["D","E","I","J","L"],
}


def _augment(mid, available, assignment, visited):
    for group in available.get(mid, []):
        assigned = next((m for m, g in assignment.items() if g == group), None)
        if assigned is None:
            assignment[mid] = group; return True
        if assigned not in visited:
            visited.add(assigned)
            if _augment(assigned, available, assignment, visited):
                assignment[mid] = group; return True
    return False


def assign_thirds(qualifying_thirds: list[dict]) -> dict[int, str]:
    q_groups  = {t["group"] for t in qualifying_thirds}
    available = {mid: [g for g in gs if g in q_groups]
                 for mid, gs in THIRD_ELIGIBLE.items()}
    match_order = sorted(available, key=lambda m: len(available[m]))
    assignment: dict[int, str] = {}
    for mid in match_order:
        _augment(mid, available, assignment, {mid})
    return assignment


def resolve_slot(slot, standings, third_assignment, thirds_by_group, match_id):
    if slot.startswith("1"):
        return standings[slot[1]].iloc[0]["team"]
    if slot.startswith("2"):
        return standings[slot[1]].iloc[1]["team"]
    if slot == "3":
        g = third_assignment.get(match_id)
        if g:
            return thirds_by_group[g]["team"]
    return "TBD"


# ---------------------------------------------------------------------------
# Fase Eliminatória
# ---------------------------------------------------------------------------

def simulate_knockouts(standings, lam_full, lam_cohost, lam_neutral, stochastic):
    qualifying_thirds = best_third_place(standings)
    thirds_by_group   = {t["group"]: t for t in qualifying_thirds}
    third_assignment  = assign_thirds(qualifying_thirds)

    mw, ml, mm, mt = {}, {}, {}, {}

    def _ko(t1, t2, mid):
        return simulate_knockout_game(t1, t2, mid, lam_full, lam_cohost, lam_neutral, stochastic)

    for match_id, slot1, slot2, _ in R32:
        t1 = resolve_slot(slot1, standings, third_assignment, thirds_by_group, match_id)
        t2 = resolve_slot(slot2, standings, third_assignment, thirds_by_group, match_id)
        mt[match_id] = (t1, t2)
        w, l, meth   = _ko(t1, t2, match_id)
        mw[match_id] = w; ml[match_id] = l; mm[match_id] = meth

    for mid, (m1, m2) in [(89,(74,77)),(90,(73,75)),(91,(76,78)),(92,(79,80)),
                           (93,(83,84)),(94,(81,82)),(95,(86,88)),(96,(85,87))]:
        t1, t2 = mw[m1], mw[m2]; mt[mid] = (t1, t2)
        w, l, meth = _ko(t1, t2, mid)
        mw[mid] = w; ml[mid] = l; mm[mid] = meth

    for mid, (m1, m2) in [(97,(89,90)),(98,(93,94)),(99,(91,92)),(100,(95,96))]:
        t1, t2 = mw[m1], mw[m2]; mt[mid] = (t1, t2)
        w, l, meth = _ko(t1, t2, mid)
        mw[mid] = w; ml[mid] = l; mm[mid] = meth

    for mid, (m1, m2) in [(101,(97,98)),(102,(99,100))]:
        t1, t2 = mw[m1], mw[m2]; mt[mid] = (t1, t2)
        w, l, meth = _ko(t1, t2, mid)
        mw[mid] = w; ml[mid] = l; mm[mid] = meth

    t1, t2 = ml[101], ml[102]; mt[103] = (t1, t2)
    w, l, meth = _ko(t1, t2, 103)
    mw[103] = w; ml[103] = l; mm[103] = meth

    t1, t2 = mw[101], mw[102]; mt[104] = (t1, t2)
    w, l, meth = _ko(t1, t2, 104)
    mw[104] = w; ml[104] = l; mm[104] = meth

    return {"winner": mw, "loser": ml, "method": mm, "teams": mt,
            "thirds": qualifying_thirds, "third_assignment": third_assignment}


# ---------------------------------------------------------------------------
# Bracket determinístico
# ---------------------------------------------------------------------------

def build_bracket_text(standings, ko, qualifying_thirds):
    lines = []
    sep   = "=" * 65

    lines.append(sep)
    lines.append("  FASE DE GRUPOS — Copa 2026")
    lines.append(sep)
    for g in sorted(GROUPS.keys()):
        df = standings[g]
        lines.append(f"\n  Grupo {g}")
        lines.append(f"  {'Pos':>3} {'Time':25s} {'Pts':>4} {'GF':>3} {'GA':>3} {'GD':>4}")
        lines.append("  " + "-" * 48)
        for _, row in df.iterrows():
            adv   = " ✓" if row["pos"] <= 2 else ""
            third = " (3º)" if row["pos"] == 3 else ""
            lines.append(f"  {int(row['pos']):3d} {row['team']:25s} {int(row['pts']):4d} "
                         f"{int(row['gf']):3d} {int(row['ga']):3d} {int(row['gd']):+4d}{adv}{third}")

    lines.append(f"\n{sep}")
    lines.append("  8 MELHORES 3ºS COLOCADOS")
    lines.append(sep)
    lines.append(f"  {'#':>2} {'Time':25s} {'Grupo':>6} {'Pts':>4} {'GD':>4} {'GF':>4}")
    lines.append("  " + "-" * 52)
    for i, t in enumerate(qualifying_thirds, 1):
        lines.append(f"  {i:2d} {t['team']:25s} {t['group']:6s} {t['pts']:4d} "
                     f"{t['gd']:+4d} {t['gf']:4d}")

    mw = ko["winner"]; mt = ko["teams"]; mm_map = ko["method"]

    def ko_round(title, match_ids):
        lines.append(f"\n{sep}"); lines.append(f"  {title}"); lines.append(sep)
        for mid in match_ids:
            t1, t2 = mt[mid]; w = mw[mid]
            ms = {"normal": "", "prorrogacao": " (prorr.)", "penaltis": " (pen.)"}[mm_map[mid]]
            lines.append(f"  M{mid:3d}: {t1:25s} vs {t2:25s}  →  {w}{ms}")

    ko_round("DEZESSEIS-AVOS (R32)", [r[0] for r in R32])
    ko_round("OITAVAS (R16)",        [89,90,91,92,93,94,95,96])
    ko_round("QUARTAS (QF)",         [97,98,99,100])
    ko_round("SEMIFINAIS",           [101,102])

    lines.append(f"\n{sep}"); lines.append("  3º LUGAR"); lines.append(sep)
    t1, t2 = mt[103]; w = mw[103]
    ms = {"normal": "", "prorrogacao": " (prorr.)", "penaltis": " (pen.)"}[mm_map[103]]
    lines.append(f"  {t1:25s} vs {t2:25s}  →  {w} (3º lugar){ms}")

    lines.append(f"\n{sep}"); lines.append("  FINAL"); lines.append(sep)
    t1, t2 = mt[104]; w = mw[104]
    ms = {"normal": "", "prorrogacao": " (prorr.)", "penaltis": " (pen.)"}[mm_map[104]]
    lines.append(f"  {t1:25s} vs {t2:25s}")
    lines.append(f"\n  *** CAMPEÃO: {w} ***{ms}")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Previsões Rodada 1
# ---------------------------------------------------------------------------

def predict_round1(model: XGBRegressor, team_feats: dict,
                   hf_home: dict, hf_cohost: dict) -> pd.DataFrame:
    """Gera predições para os 8 jogos da 1ª rodada (Jun 11-13/2026)."""
    from scipy.stats import poisson as sp_poisson

    def poisson_probs(lh, la, max_g=10):
        p_w = p_d = p_l = 0.0
        for h in range(max_g + 1):
            ph = sp_poisson.pmf(h, lh)
            for a in range(max_g + 1):
                p = ph * sp_poisson.pmf(a, la)
                if h > a:    p_w += p
                elif h == a: p_d += p
                else:        p_l += p
        return p_w, p_d, p_l

    def top3_scores(lh, la, max_g=8):
        scores = []
        for h in range(max_g + 1):
            ph = sp_poisson.pmf(h, lh)
            for a in range(max_g + 1):
                p = ph * sp_poisson.pmf(a, la)
                scores.append((h, a, p))
        scores.sort(key=lambda x: x[2], reverse=True)
        return [f"{h}-{a}" for h, a, _ in scores[:3]]

    def build_feature_row(team, opponent, venue_country):
        f1 = team_feats.get(team, {})
        f2 = team_feats.get(opponent, {})
        hf = get_host_factor(team, venue_country, hf_home, hf_cohost)
        row = {
            "opp_saves_decay":     f2.get("saves_decay", 2.0),
            "shots_on_goal_decay": f1.get("shots_on_goal_decay", 3.5),
            "gols_sofridos_decay": f1.get("gols_sofridos_decay", np.nan),
            "host_factor":         hf,
            "conf_AFC": 0, "conf_CAF": 0, "conf_CONCACAF": 0,
            "conf_CONMEBOL": 0, "conf_OFC": 0, "conf_UEFA": 0,
        }
        for col in RATING_FEATURES:
            if col == "elo_diff_decay":
                elo1 = f1.get("_elo_current", np.nan)
                elo2 = f2.get("_elo_current", np.nan)
                row[col] = (float(elo1) - float(elo2)
                            if pd.notna(elo1) and pd.notna(elo2)
                            else f1.get(col, np.nan))
            else:
                row[col] = f1.get(col, np.nan)
        return row

    records = []
    for t1, t2, date_str, group, venue in ROUND1_GAMES:
        row_h = build_feature_row(t1, t2, venue)
        row_a = build_feature_row(t2, t1, venue)
        df_pred = pd.DataFrame([row_h, row_a])
        lams = model.predict(df_pred[FEATURES])
        lh, la = float(lams[0]), float(lams[1])
        pw, pd_, pl = poisson_probs(lh, la)
        sc = top3_scores(lh, la)
        records.append({
            "jogo":          f"{t1} vs {t2}",
            "data":          date_str,
            "grupo":         group,
            "venue":         venue,
            "prob_V":        round(pw * 100, 1),
            "prob_E":        round(pd_ * 100, 1),
            "prob_D":        round(pl * 100, 1),
            "lambda_home":   round(lh, 3),
            "lambda_away":   round(la, 3),
            "placar_1":      sc[0],
            "placar_2":      sc[1],
            "placar_3":      sc[2],
            "host_factor_t1": round(get_host_factor(t1, venue, hf_home, hf_cohost), 4),
            "host_factor_t2": round(get_host_factor(t2, venue, hf_home, hf_cohost), 4),
        })

    df_r1 = pd.DataFrame(records)
    df_r1.to_csv(OUT_R1, index=False, encoding="utf-8")
    return df_r1


def print_round1_predictions(df_r1: pd.DataFrame) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  PREVISÕES — Copa 2026 RODADA 1 (v10, venue-aware host_factor)")
    print(sep)
    print(f"  {'Jogo':35s} {'Data':>11} {'V%':>6} {'E%':>6} {'D%':>6} {'λH':>6} {'λA':>6}  Placar 1     HF_H  HF_A")
    print("  " + "-" * 87)
    for _, r in df_r1.iterrows():
        jogo = r["jogo"][:34]
        print(f"  {jogo:35s} {r['data']:>11} "
              f"{r['prob_V']:6.1f} {r['prob_E']:6.1f} {r['prob_D']:6.1f} "
              f"{r['lambda_home']:6.3f} {r['lambda_away']:6.3f}  "
              f"{r['placar_1']:10s}  {r['host_factor_t1']:5.4f} {r['host_factor_t2']:5.4f}")
    print(sep)
    print(f"  Salvo: {OUT_R1}")


# ---------------------------------------------------------------------------
# Monte Carlo com CI
# ---------------------------------------------------------------------------

STAGE_LABELS = ["Fase Grupos", "Oitavas", "Quartas", "Semi", "Final", "Campeão"]


def run_monte_carlo_ci(lam_full, lam_cohost, lam_neutral,
                       n: int = N_SIM) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    all_teams   = [t for ts in GROUPS.values() for t in ts]
    FOCUS_TEAMS = {"Brazil", "Argentina", "France", "Spain", "Germany",
                   "Portugal", "Japan", "Norway", "Morocco"}

    counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in STAGE_KEY}
    focus_rounds: dict[str, list[int]] = {t: [] for t in FOCUS_TEAMS}

    print(f"  Rodando {n:,} simulações Monte Carlo...")
    step = max(1, n // 10)

    for sim_i in range(n):
        if sim_i % step == 0:
            pct = sim_i * 100 // n
            print(f"    {pct:3d}%", end="\r", flush=True)

        standings = simulate_all_groups(lam_full, lam_cohost, lam_neutral, stochastic=True)
        q_thirds  = best_third_place(standings)

        sim_max: dict[str, int] = {t: -1 for t in FOCUS_TEAMS}

        for g, df in standings.items():
            for _, row in df.iterrows():
                if row["pos"] <= 2:
                    counts["grupo"][row["team"]] += 1
                    if row["team"] in FOCUS_TEAMS:
                        sim_max[row["team"]] = max(sim_max[row["team"]], 0)
        for t in q_thirds:
            counts["grupo"][t["team"]] += 1
            if t["team"] in FOCUS_TEAMS:
                sim_max[t["team"]] = max(sim_max[t["team"]], 0)

        ko = simulate_knockouts(standings, lam_full, lam_cohost, lam_neutral, stochastic=True)
        mw = ko["winner"]; mt = ko["teams"]

        for mid, _, _, _ in R32:
            counts["r32"][mw[mid]] += 1
            if mw[mid] in FOCUS_TEAMS:
                sim_max[mw[mid]] = max(sim_max[mw[mid]], 1)
        for mid in [89,90,91,92,93,94,95,96]:
            counts["r16"][mw[mid]] += 1
            if mw[mid] in FOCUS_TEAMS:
                sim_max[mw[mid]] = max(sim_max[mw[mid]], 2)
        for mid in [97,98,99,100]:
            counts["qf"][mw[mid]] += 1
            if mw[mid] in FOCUS_TEAMS:
                sim_max[mw[mid]] = max(sim_max[mw[mid]], 3)
        for mid in [101,102]:
            for t in mt[mid]:
                counts["sf"][t] += 1
            if mw[mid] in FOCUS_TEAMS:
                sim_max[mw[mid]] = max(sim_max[mw[mid]], 4)
        counts["3rd_place"][mw[103]] += 1
        counts["champion"][mw[104]] += 1
        if mw[104] in FOCUS_TEAMS:
            sim_max[mw[104]] = max(sim_max[mw[104]], 5)

        for t in FOCUS_TEAMS:
            focus_rounds[t].append(sim_max[t])

    print(f"    100% — concluído.")

    rows = []
    for team in sorted(all_teams):
        row = {"team": team, "group": TEAM_TO_GROUP.get(team, "?")}
        for stage in STAGE_KEY:
            row[stage] = round(counts[stage].get(team, 0) / n * 100, 2)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)
    return df, focus_rounds


# ---------------------------------------------------------------------------
# Tabelas de resultados
# ---------------------------------------------------------------------------

def print_probs_table(mc_df: pd.DataFrame) -> None:
    sep = "=" * 95
    print(f"\n{sep}")
    print("  PROBABILIDADES MONTE CARLO — Copa 2026 (v10: venue-aware host_factor)")
    print(f"  {N_SIM:,} simulações  |  XGBoost count:poisson  |  CV até {CUTOFF}")
    print(sep)
    print(f"  {'Time':25s} {'Grp':>4} {'Grupos%':>8} {'Oitavos%':>9} {'Quartas%':>9} "
          f"{'Semi%':>7} {'Final%':>7} {'3ºLugar%':>9} {'Campeão%':>9}")
    print("  " + "-" * 91)

    for _, row in mc_df.iterrows():
        if row["champion"] < 0.05 and row["sf"] < 1.0:
            continue
        print(f"  {row['team']:25s} {row['group']:>4} "
              f"{row['grupo']:8.1f} {row['r32']:9.1f} {row['r16']:9.1f} "
              f"{row['qf']:7.1f} {row['sf']:7.1f} {row['3rd_place']:9.1f} "
              f"{row['champion']:9.1f}")
    print(sep)

    print("\n  TOP-10 CAMPEÕES MAIS PROVÁVEIS (v10):")
    print("  " + "-" * 40)
    for rank, (_, row) in enumerate(mc_df.head(10).iterrows(), 1):
        bar = "█" * max(1, int(row["champion"] / mc_df["champion"].max() * 25))
        print(f"  {rank:2d}. {row['team']:22s} {row['champion']:6.1f}%  {bar}")
    print(sep)


def _print_top15(mc_df: pd.DataFrame) -> None:
    sep   = "=" * 75
    print(f"\n{sep}")
    print("  TOP-15 CAMPEÕES — Copa 2026 (v10: venue-aware host_factor)")
    print(sep)
    print(f"  {'#':>3}  {'Time':22s}  {'Campeão%':>9}  {'Final%':>7}  {'Semi%':>6}  {'Barra'}")
    print("  " + "-" * 72)
    top15    = mc_df.head(15)
    max_champ = top15["champion"].max() or 1.0
    for rank, (_, row) in enumerate(top15.iterrows(), 1):
        bar = "█" * max(1, int(row["champion"] / max_champ * 28))
        print(f"  {rank:3d}  {row['team']:22s}  {row['champion']:9.1f}%  "
              f"{row['sf']:7.1f}%  {row['qf']:6.1f}%  {bar}")
    print(sep)


def print_comparison_with_prev(mc_new: pd.DataFrame) -> None:
    if not OUT_MC_PREV.exists():
        print(f"\n  [comparação] Arquivo anterior não encontrado: {OUT_MC_PREV}")
        return

    mc_old = pd.read_csv(OUT_MC_PREV)
    sep    = "=" * 80
    champ_col_old = "champion" if "champion" in mc_old.columns else mc_old.columns[-1]
    focus = ["Brazil", "Argentina", "France", "Spain", "Germany",
             "Portugal", "Japan", "Norway", "Morocco", "Netherlands",
             "Mexico", "United States", "Canada"]

    print(f"\n{sep}")
    print("  COMPARAÇÃO: v9 (opp_saves + static hf) → v10 (venue-aware hf, sem opp_saves)")
    print(sep)
    print(f"  {'Time':22s} {'v9 Campeão%':>13} {'v10 Campeão%':>13} {'Δ':>8}")
    print("  " + "-" * 60)

    for team in focus:
        row_new = mc_new[mc_new["team"] == team]
        row_old = mc_old[mc_old["team"] == team]
        c_new = float(row_new["champion"].values[0]) if not row_new.empty else 0.0
        c_old = float(row_old[champ_col_old].values[0]) if not row_old.empty else 0.0
        delta = c_new - c_old
        arrow = "▲" if delta > 0.5 else ("▼" if delta < -0.5 else " ")
        print(f"  {team:22s} {c_old:13.1f}% {c_new:13.1f}%  {arrow}{delta:+.1f}pp")
    print(sep)


def _print_focus_ci(mc_df: pd.DataFrame, focus_rounds: dict[str, list[int]]) -> None:
    focus = ["Brazil", "Argentina", "France", "Spain", "Germany",
             "Portugal", "Japan", "Norway"]
    sep   = "=" * 78
    print(f"\n{sep}")
    print("  FOCO — Times prioritários (v10)")
    print(f"  Intervalo de confiança percentil 5–95 por fase")
    print(sep)
    print(f"  {'Time':15s} {'Campeão%':>9}  {'p5-fase':>11}  {'p50-fase':>12}  {'p95-fase':>12}")
    print("  " + "-" * 64)
    for team in focus:
        champ_row = mc_df.loc[mc_df["team"] == team, "champion"]
        champ     = float(champ_row.values[0]) if not champ_row.empty else 0.0
        rounds    = focus_rounds.get(team, [])
        if rounds:
            p5  = STAGE_LABELS[max(0, min(5, int(np.percentile(rounds, 5)  + 1)))]
            p50 = STAGE_LABELS[max(0, min(5, int(np.percentile(rounds, 50) + 1)))]
            p95 = STAGE_LABELS[max(0, min(5, int(np.percentile(rounds, 95) + 1)))]
        else:
            p5 = p50 = p95 = "N/A"
        print(f"  {team:15s} {champ:9.1f}%  {p5:>11}  {p50:>12}  {p95:>12}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 65
    print(f"\n{sep}")
    print("  simulate_copa2026.py — Copa do Mundo 2026")
    print("  Modelo v10: venue-aware host_factor  |  opp_saves_decay restaurado  |  16 features")
    print(f"  Treino até {CUTOFF}  |  {N_SIM:,} simulações Monte Carlo")
    print(sep)

    print("\n── Treinando modelo e extraindo features... ──")
    model, team_feats = train_model()

    # Carregar host factors do CSV
    hf_home, hf_cohost = load_host_factors()

    all_teams = [t for ts in GROUPS.values() for t in ts]
    missing   = [t for t in all_teams if t not in team_feats]
    if missing:
        print(f"  AVISO: {len(missing)} times sem features — usando defaults")
        for t in missing:
            team_feats[t] = {"shots_on_goal_decay": 3.0, "gols_sofridos_decay": np.nan}
    print(f"  {len(all_teams)} times cobertos  |  {len(FEATURES)} features")

    # Validação host_factor
    validate_host_factors(hf_home, hf_cohost)

    print("\n── Pré-computando lambdas (3 caches: full / cohost / neutral)... ──")
    lam_full    = precompute_lambdas(model, team_feats, all_teams, hf_home)
    lam_cohost  = precompute_lambdas(model, team_feats, all_teams, hf_cohost)
    lam_neutral = precompute_lambdas(model, team_feats, all_teams, {})
    print(f"  {len(lam_full)} pares × 3 caches pré-computados")

    # Previsões Rodada 1
    print("\n── Previsões Rodada 1 (11-13/jun) ──")
    df_r1 = predict_round1(model, team_feats, hf_home, hf_cohost)
    print_round1_predictions(df_r1)

    # Bracket determinístico
    print("\n── Simulação determinística (modo Poisson) ──")
    np.random.seed(SEED)
    det_standings = simulate_all_groups(lam_full, lam_cohost, lam_neutral, stochastic=False)
    det_ko        = simulate_knockouts(det_standings, lam_full, lam_cohost, lam_neutral, stochastic=False)
    q_thirds      = best_third_place(det_standings)
    bracket_txt   = build_bracket_text(det_standings, det_ko, q_thirds)
    print(bracket_txt)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(bracket_txt)
    print(f"\n  Bracket salvo: {OUT_TXT}")

    champion = det_ko["winner"][104]
    print(f"\n  Bracket det. → Campeão: {champion}")

    # Monte Carlo
    print(f"\n── Monte Carlo ({N_SIM:,} simulações) ──")
    mc_df, focus_rounds = run_monte_carlo_ci(lam_full, lam_cohost, lam_neutral, n=N_SIM)

    mc_target = OUT_MC
    try:
        mc_df.to_csv(mc_target, index=False, encoding="utf-8")
    except PermissionError:
        mc_target = OUT_MC.with_stem(OUT_MC.stem + "_v2")
        mc_df.to_csv(mc_target, index=False, encoding="utf-8")
    print(f"  Resultados salvos: {mc_target}")

    print_probs_table(mc_df)
    print_comparison_with_prev(mc_df)
    _print_focus_ci(mc_df, focus_rounds)
    _print_top15(mc_df)


if __name__ == "__main__":
    main()
