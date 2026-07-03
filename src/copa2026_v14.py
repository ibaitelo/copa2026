#!/usr/bin/env python3
"""
copa2026_v14.py — Pipeline atualizado: R32 resultados reais + simulação restante

Mudanças vs v13:
- CUTOFF = 2026-07-03 (inclui 10 jogos R32 já finalizados no treino)
- Resultados reais dos 10 jogos R32 já jogados
- Belgium vs Senegal (correção vs v13 que tinha Belgium vs Algeria)
- Switzerland vs Algeria (real, já no DB)
- Simulação determinística: usa resultado real onde disponível, modelo onde não
- MC 10k a partir dos 6 jogos R32 pendentes (bracket já parcialmente resolvido)

Jogos R32 já disputados (resultados reais):
  South Africa  0-1  Canada        (28/Jun)
  Brazil        2-1  Japan         (29/Jun)  ← Brasil ganhou! vs previsão anterior
  Ivory Coast   1-2  Norway        (30/Jun)
  France        3-0  Sweden        (30/Jun)
  Mexico        2-0  Ecuador       (01/Jul)
  England       2-1  DR Congo      (01/Jul)
  United States 2-0  Bosnia        (02/Jul)
  Spain         3-0  Austria       (02/Jul)
  Portugal      2-1  Croatia       (02/Jul)
  Switzerland   2-0  Algeria       (03/Jul)  ← último jogo salvo

Jogos R32 pendentes (a simular pelo modelo):
  Germany       vs   Paraguay
  Netherlands   vs   Morocco
  Belgium       vs   Senegal
  Argentina     vs   Cape Verde
  Colombia      vs   Ghana
  Australia     vs   Egypt
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson
from xgboost import XGBRegressor
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_CSV   = Path("data/raw/full_dataset_raw.csv")
MODEL_CSV = Path("data/processed/model_dataset.csv")
ELO_CSV   = Path("data/raw/elo_history.csv")
OUT_PKL   = Path("outputs/xgboost_v14.pkl")
OUT_XLSX  = Path("outputs/copa2026_full_v14.xlsx")

CUTOFF = pd.Timestamp("2026-07-03")
SEED   = 42
N_SIM  = 10_000

CONF_DUMMIES = ["conf_AFC","conf_CAF","conf_CONCACAF","conf_CONMEBOL","conf_OFC","conf_UEFA"]
RATING_FEATURES = [
    "elo_diff_decay","delta_rating_decay","gls_ponderado_decay",
    "gls_decay_vs_forte","gls_decay_vs_fraco","margem_gols_decay",
]
FEATURES = (["opp_saves_decay","shots_on_goal_decay","gols_sofridos_decay"]
            + RATING_FEATURES + CONF_DUMMIES)

XGB_PARAMS = dict(
    objective="count:poisson", n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, random_state=42, n_jobs=-1, verbosity=0,
)

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
ALL_COPA_TEAMS = [t for ts in GROUPS.values() for t in ts]

COPA_TEAM_CONF: dict[str, str] = {
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

CONF_MEDIANS_SAVES = {
    "UEFA":2.9353,"CONMEBOL":2.7985,"AFC":2.4974,
    "CONCACAF":2.4799,"CAF":1.8144,"OFC":2.4974,
}
GLOBAL_MEDIAN_SAVES = 2.6

NAME_MAP = {
    "Czechia":"Czech Republic","Czech Rep.":"Czech Republic",
    "Côte d'Ivoire":"Ivory Coast","Cote d'Ivoire":"Ivory Coast",
    "USA":"United States","IR Iran":"Iran",
    "Korea Republic":"South Korea","Bosnia & Herzegovina":"Bosnia and Herzegovina",
    "Türkiye":"Turkey","Cape Verde Islands":"Cape Verde","Congo DR":"DR Congo",
    "México":"Mexico","Curaçao":"Curacao",
}

# ── R32 COMPLETO: resultados reais + matchups restantes ───────────────────────
# (mid, t1, t2, score_real_g1|None, score_real_g2|None, went_penalties)
R32_FULL = [
    # Jogos DISPUTADOS — resultados reais da API
    (73,  "South Africa",          "Canada",                   0, 1, False),
    (74,  "Germany",               "Paraguay",                 None, None, None),  # pendente
    (75,  "Netherlands",           "Morocco",                  None, None, None),  # pendente
    (76,  "Brazil",                "Japan",                    2, 1, False),  # Brasil ganhou!
    (77,  "France",                "Sweden",                   3, 0, False),
    (78,  "Ivory Coast",           "Norway",                   1, 2, False),
    (79,  "Mexico",                "Ecuador",                  2, 0, False),
    (80,  "England",               "DR Congo",                 2, 1, False),
    (81,  "United States",         "Bosnia and Herzegovina",   2, 0, False),
    (82,  "Belgium",               "Senegal",                  None, None, None),  # pendente (Belgium vs Senegal — não Algeria)
    (83,  "Portugal",              "Croatia",                  2, 1, False),
    (84,  "Spain",                 "Austria",                  3, 0, False),
    (85,  "Switzerland",           "Algeria",                  2, 0, False),  # resultado real
    (86,  "Argentina",             "Cape Verde",               None, None, None),  # pendente
    (87,  "Colombia",              "Ghana",                    None, None, None),  # pendente
    (88,  "Australia",             "Egypt",                    None, None, None),  # pendente
]

# Vencedores já conhecidos do R32 (resultados reais)
KNOWN_R32_WINNERS = {
    73: "Canada",
    76: "Brazil",
    77: "France",
    78: "Norway",
    79: "Mexico",
    80: "England",
    81: "United States",
    83: "Portugal",
    84: "Spain",
    85: "Switzerland",
}

# QF: winner(mid1) vs winner(mid2) — bracket fixo
QF_BRACKET = [
    (89,  73, 74),
    (90,  75, 76),
    (91,  77, 78),
    (92,  79, 80),
    (93,  81, 82),
    (94,  83, 84),
    (95,  85, 86),
    (96,  87, 88),
]
SF_BRACKET = [
    (97,  89, 90),
    (98,  91, 92),
    (99,  93, 94),
    (100, 95, 96),
]
FINAL_BRACKET = [
    (103, 97, 98),
    (104, 99, 100),
]

SEP = "=" * 72


# =============================================================================
# 1. Dados e treino
# =============================================================================

def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in wide.iterrows():
        for side, opp in [("home","away"),("away","home")]:
            row: dict = {
                "date":                r["date"],
                "team":                r[f"{side}_team"],
                "gols_marcados":       r[f"{side}_gols"],
                "match_type":          r.get("match_type", ""),
                "opp_saves_decay":     float(r.get(f"{opp}_saves_decay", 1.0)),
                "shots_on_goal_decay": float(r[f"{side}_shots_on_goal_decay"]),
                "gols_sofridos_decay": float(r.get(f"{side}_gols_sofridos_decay", np.nan)),
                "saves_decay":         float(r.get(f"{side}_saves_decay", 1.0)),
            }
            for col in CONF_DUMMIES:
                row[col] = int(r.get(col, 0))
            for col in RATING_FEATURES:
                v = r.get(f"{side}_{col}", np.nan)
                row[col] = float(v) if pd.notna(v) else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def apply_saves_norm(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.copy()
    for side, opp in [("home","away"),("away","home")]:
        col = f"{side}_saves_decay"
        if col not in wide.columns:
            continue
        opp_col = f"{opp}_team"
        med = wide[opp_col].map(
            lambda t: CONF_MEDIANS_SAVES.get(COPA_TEAM_CONF.get(t, "UEFA"), GLOBAL_MEDIAN_SAVES)
        ).fillna(GLOBAL_MEDIAN_SAVES)
        wide[col] = wide[col].astype(float) / med
    return wide


def winsorize(long: pd.DataFrame) -> pd.DataFrame:
    feats = ["shots_on_goal_decay","gols_sofridos_decay","opp_saves_decay",
             "gls_ponderado_decay","gls_decay_vs_forte","gls_decay_vs_fraco",
             "margem_gols_decay","delta_rating_decay"]
    long = long.copy()
    for feat in feats:
        if feat not in long.columns:
            continue
        col = long[feat].dropna()
        p1, p99 = col.quantile(0.01), col.quantile(0.99)
        long[feat] = long[feat].clip(lower=p1, upper=p99)
    return long


def train_model(long: pd.DataFrame, cutoff: pd.Timestamp) -> XGBRegressor:
    tr = long[long["date"] <= cutoff].copy()
    X  = tr[FEATURES].fillna(tr[FEATURES].median())
    y  = tr["gols_marcados"]
    m  = XGBRegressor(**XGB_PARAMS)
    m.fit(X, y)
    yp   = m.predict(X)
    mae  = np.mean(np.abs(yp - y))
    bias = yp.mean() - y.mean()
    print(f"  n={len(tr):,}  MAE={mae:.4f}  bias={bias:+.4f}")
    wc = tr["date"] >= pd.Timestamp("2026-06-28")
    if wc.sum() > 0:
        print(f"  WC (GS+R32) n={wc.sum()}  MAE={np.mean(np.abs(yp[wc]-y.values[wc])):.4f}"
              f"  bias={yp[wc].mean()-y.values[wc].mean():+.4f}")
    return m


# =============================================================================
# 2. Features e lambdas
# =============================================================================

def extract_feats(long: pd.DataFrame) -> dict[str, dict]:
    tf: dict[str, dict] = {}
    for _, row in long.sort_values("date").iterrows():
        entry: dict = {
            "saves_decay":         float(row.get("saves_decay", 1.0)),
            "shots_on_goal_decay": float(row.get("shots_on_goal_decay", 3.5)),
            "gols_sofridos_decay": row.get("gols_sofridos_decay", np.nan),
        }
        for col in RATING_FEATURES:
            v = row.get(col, np.nan)
            entry[col] = float(v) if pd.notna(v) else np.nan
        tf[row["team"]] = entry
    if ELO_CSV.exists():
        elo = pd.read_csv(ELO_CSV, parse_dates=["date"])
        for team, val in elo.sort_values("date").groupby("team")["elo_after"].last().items():
            tf.setdefault(team, {})["_elo_current"] = float(val)
    return tf


def compute_lambdas(model, tf: dict, teams: list[str]) -> dict[tuple[str,str], float]:
    rows, pairs = [], []
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            f1, f2 = tf.get(t1, {}), tf.get(t2, {})
            row: dict = {
                "opp_saves_decay":     f2.get("saves_decay", 1.0),
                "shots_on_goal_decay": f1.get("shots_on_goal_decay", 3.5),
                "gols_sofridos_decay": f1.get("gols_sofridos_decay", np.nan),
                **{c: 0 for c in CONF_DUMMIES},
            }
            for col in RATING_FEATURES:
                if col == "elo_diff_decay":
                    e1 = f1.get("_elo_current", np.nan)
                    e2 = f2.get("_elo_current", np.nan)
                    row[col] = float(e1)-float(e2) if (pd.notna(e1) and pd.notna(e2)) else f1.get(col, np.nan)
                else:
                    row[col] = f1.get(col, np.nan)
            c = COPA_TEAM_CONF.get(t1, "")
            ck = f"conf_{c}"
            if ck in CONF_DUMMIES:
                row[ck] = 1
            rows.append(row)
            pairs.append((t1, t2))
    preds = model.predict(pd.DataFrame(rows)[FEATURES])
    return {p: float(v) for p, v in zip(pairs, preds)}


# =============================================================================
# 3. Poisson utils
# =============================================================================

def best_score(l1: float, l2: float, max_g: int = 8) -> tuple[int, int, float]:
    bp, bg1, bg2 = -1.0, 0, 0
    for g1 in range(max_g+1):
        for g2 in range(max_g+1):
            p = scipy_poisson.pmf(g1, l1) * scipy_poisson.pmf(g2, l2)
            if p > bp:
                bp, bg1, bg2 = p, g1, g2
    return bg1, bg2, bp


def match_probs(l1: float, l2: float, max_g: int = 8) -> tuple[float, float, float]:
    p1 = pd = p2 = 0.0
    for g1 in range(max_g+1):
        for g2 in range(max_g+1):
            p = scipy_poisson.pmf(g1, l1) * scipy_poisson.pmf(g2, l2)
            if g1 > g2:    p1 += p
            elif g1 == g2: pd += p
            else:          p2 += p
    return p1, pd, p2


def knock_prob(l1: float, l2: float) -> float:
    p1, pd, p2 = match_probs(l1, l2)
    return p1 + 0.5 * pd


def sim_ko(l1: float, l2: float, rng: np.random.Generator) -> int:
    g1, g2 = rng.poisson(l1), rng.poisson(l2)
    if g1 > g2:   return 0
    elif g2 > g1: return 1
    else:         return int(rng.random() > 0.5)


# =============================================================================
# 4. Simulação determinística
# =============================================================================

def simulate_deterministic(lam: dict) -> dict:
    winners: dict[int, str] = {}
    games:   list[dict] = []

    def play(mid: int, t1: str, t2: str, stage: str,
             real_g1=None, real_g2=None, real_pen=False) -> str:
        l1 = lam.get((t1, t2), 1.3)
        l2 = lam.get((t2, t1), 1.3)
        p1, pd_, p2 = match_probs(l1, l2)
        pk = knock_prob(l1, l2)

        if real_g1 is not None:
            # Resultado real
            g1, g2 = real_g1, real_g2
            if g1 > g2:    winner, note, is_real = t1, "", True
            elif g2 > g1:  winner, note, is_real = t2, "", True
            elif real_pen:
                winner = t1 if pk >= 0.5 else t2
                note, is_real = " (pên)", True
            else:          winner, note, is_real = t1, " (pên)", True
        else:
            # Predição do modelo
            g1, g2, _ = best_score(l1, l2)
            is_real = False
            if g1 > g2:    winner, note = t1, ""
            elif g2 > g1:  winner, note = t2, ""
            else:          winner, note = (t1 if l1 >= l2 else t2), " (pên)"

        games.append({
            "stage": stage, "mid": mid,
            "t1": t1, "t2": t2,
            "g1": g1, "g2": g2,
            "score": f"{g1}–{g2}{note}",
            "winner": winner,
            "l1": round(l1, 3), "l2": round(l2, 3),
            "p1": round(p1, 4), "pd": round(pd_, 4), "p2": round(p2, 4),
            "pk": round(pk, 4),
            "is_real": is_real,
        })
        return winner

    # R32 — mistura de resultados reais e previsões
    r32_ids = [mid for mid, *_ in R32_FULL]
    for mid, t1, t2, rg1, rg2, rpen in R32_FULL:
        winner = play(mid, t1, t2, "R32", rg1, rg2, rpen)
        winners[mid] = winner

    # QF
    for qf_id, r32a, r32b in QF_BRACKET:
        t1, t2 = winners[r32a], winners[r32b]
        winners[qf_id] = play(qf_id, t1, t2, "QF")

    # SF
    for sf_id, qfa, qfb in SF_BRACKET:
        t1, t2 = winners[qfa], winners[qfb]
        winners[sf_id] = play(sf_id, t1, t2, "SF")

    # 3rd place
    def loser(mid: int) -> str:
        g = next(gm for gm in games if gm["mid"] == mid)
        return g["t2"] if g["winner"] == g["t1"] else g["t1"]

    play(101, loser(97), loser(98), "3o Lugar A")
    play(102, loser(99), loser(100), "3o Lugar B")

    # Finals
    play(103, winners[97], winners[98], "FINAL A")
    play(104, winners[99], winners[100], "FINAL B")

    return {"games": games, "winners": winners}


# =============================================================================
# 5. Monte Carlo
# =============================================================================

def run_mc(lam: dict) -> tuple[dict, dict]:
    rng = np.random.default_rng(SEED)
    champ: dict[str, int] = {}
    radv: dict[str, dict[str, int]] = {
        t: {"R32": 0, "QF": 0, "SF": 0, "Final": 0, "Campeon": 0}
        for t in ALL_COPA_TEAMS
    }

    r32_ids = [mid for mid, *_ in R32_FULL]

    for _ in range(N_SIM):
        w: dict[int, str] = {}

        # R32
        for mid, t1, t2, rg1, rg2, rpen in R32_FULL:
            if rg1 is not None:
                # Resultado já conhecido
                if rg1 > rg2:    winner = t1
                elif rg2 > rg1:  winner = t2
                else:            winner = t1 if lam.get((t1,t2),1.3) >= lam.get((t2,t1),1.3) else t2
            else:
                l1, l2 = lam.get((t1, t2), 1.3), lam.get((t2, t1), 1.3)
                idx = sim_ko(l1, l2, rng)
                winner = t1 if idx == 0 else t2
            w[mid] = winner
            radv[winner]["R32"] += 1

        # QF
        for qf_id, r32a, r32b in QF_BRACKET:
            t1, t2 = w[r32a], w[r32b]
            l1, l2 = lam.get((t1, t2), 1.3), lam.get((t2, t1), 1.3)
            idx = sim_ko(l1, l2, rng)
            w[qf_id] = t1 if idx == 0 else t2
            radv[w[qf_id]]["QF"] += 1

        # SF
        for sf_id, qfa, qfb in SF_BRACKET:
            t1, t2 = w[qfa], w[qfb]
            l1, l2 = lam.get((t1, t2), 1.3), lam.get((t2, t1), 1.3)
            idx = sim_ko(l1, l2, rng)
            w[sf_id] = t1 if idx == 0 else t2
            radv[w[sf_id]]["SF"] += 1

        # Finals
        for fin_id, sf1, sf2 in FINAL_BRACKET:
            t1, t2 = w[sf1], w[sf2]
            l1, l2 = lam.get((t1, t2), 1.3), lam.get((t2, t1), 1.3)
            radv[t1]["Final"] += 1; radv[t2]["Final"] += 1
            idx = sim_ko(l1, l2, rng)
            winner = t1 if idx == 0 else t2
            w[fin_id] = winner
            radv[winner]["Campeon"] += 1
            champ[winner] = champ.get(winner, 0) + 1

    return champ, radv


# =============================================================================
# 6. Standings reais
# =============================================================================

def compute_standings(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    copa = raw[(raw["match_type"] == "WC") & (raw["home_away"] == "home")].copy()
    copa["date"] = pd.to_datetime(copa["date"])

    team_to_group = {t: g for g, ts in GROUPS.items() for t in ts}
    s: dict[str, dict[str, dict]] = {
        g: {t: {"pts":0,"gf":0,"ga":0,"gd":0,"w":0,"d":0,"l":0,"pgj":0}
            for t in ts}
        for g, ts in GROUPS.items()
    }

    for _, row in copa.iterrows():
        home = NAME_MAP.get(row["team"], row["team"])
        away = NAME_MAP.get(row["opponent"], row["opponent"])
        gh = pd.to_numeric(row["gols_marcados"], errors="coerce")
        ga = pd.to_numeric(row["gols_sofridos"], errors="coerce")
        if pd.isna(gh) or pd.isna(ga):
            continue
        gh, ga = int(gh), int(ga)
        grp = team_to_group.get(home)
        if grp is None or team_to_group.get(away) != grp:
            continue
        for team, gf, gc in [(home, gh, ga), (away, ga, gh)]:
            if team not in s[grp]:
                continue
            d = s[grp][team]
            d["gf"] += gf; d["ga"] += gc; d["gd"] = d["gf"] - d["ga"]; d["pgj"] += 1
            if gf > gc:    d["pts"] += 3; d["w"] += 1
            elif gf == gc: d["pts"] += 1; d["d"] += 1
            else:          d["l"] += 1

    result: dict[str, pd.DataFrame] = {}
    for g, td in s.items():
        df = pd.DataFrame([{"team": t, **v} for t, v in td.items()])
        df = df.sort_values(["pts","gd","gf"], ascending=[False,False,False]).reset_index(drop=True)
        df["pos"] = df.index + 1
        result[g] = df
    return result


# =============================================================================
# 7. Excel
# =============================================================================

def _hdr(ws, r, c, val, bg="1A5276", fg="FFFFFF", bold=True, size=11):
    cell = ws.cell(row=r, column=c, value=val)
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.font = Font(bold=bold, color=fg, size=size)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    return cell


def _cell(ws, r, c, val, bg=None, bold=False, fmt=None, align="center"):
    cell = ws.cell(row=r, column=c, value=val)
    if bg:
        cell.fill = PatternFill("solid", fgColor=bg)
    cell.font = Font(bold=bold, size=10)
    cell.alignment = Alignment(horizontal=align, vertical="center")
    if fmt:
        cell.number_format = fmt
    s = Side(border_style="thin", color="CCCCCC")
    cell.border = Border(left=s, right=s, top=s, bottom=s)
    return cell


STAGE_BG = {
    "R32": "D6EAF8", "QF": "D5F5E3", "SF": "FDEBD0",
    "3o Lugar A": "E8DAEF", "3o Lugar B": "E8DAEF",
    "FINAL A": "FDEDEC", "FINAL B": "FDEDEC",
}


def build_excel(out: Path, standings, det, mc_champ, radv, lam):
    wb = openpyxl.Workbook()

    # ── Aba 1: Grupos ──────────────────────────────────────────────────────────
    ws1 = wb.active; ws1.title = "Grupos"; ws1.sheet_view.showGridLines = False
    hdrs = ["Pos","Time","PJ","V","E","D","GP","GC","SG","Pts"]
    row = 1
    for g in sorted(GROUPS.keys()):
        _hdr(ws1, row, 1, f"Grupo {g}", bg="1A5276"); ws1.merge_cells(f"A{row}:J{row}"); row += 1
        for c, h in enumerate(hdrs, 1): _hdr(ws1, row, c, h, bg="2E86C1"); row += 1
        for _, r in standings[g].iterrows():
            bg = "FDFEFE" if int(r["pos"]) % 2 == 0 else "EBF5FB"
            medal = {1:"🥇",2:"🥈",3:"🥉"}.get(int(r["pos"]),"  ")
            vals = [f"{medal} {int(r['pos'])}",r["team"],int(r["pgj"]),int(r["w"]),
                    int(r["d"]),int(r["l"]),int(r["gf"]),int(r["ga"]),int(r["gd"]),int(r["pts"])]
            for c, v in enumerate(vals, 1):
                _cell(ws1, row, c, v, bg=bg, bold=(int(r["pos"])<=2),
                      align="left" if c==2 else "center")
            row += 1
        row += 1
    ws1.column_dimensions["B"].width = 26
    for cl in list("ACDEFGHIJ"): ws1.column_dimensions[cl].width = 7

    # ── Aba 2: Bracket Determinístico ─────────────────────────────────────────
    ws2 = wb.create_sheet("Bracket"); ws2.sheet_view.showGridLines = False
    _hdr(ws2, 1, 1, "Simulação Copa 2026 — Bracket Completo (Resultado Real + Previsão Modelo)", bg="154360")
    ws2.merge_cells("A1:J1")
    hdrs2 = ["Stage","#","Tipo","Time 1","λ1","Placar","λ2","Time 2","P(adv T1)","Vencedor"]
    for c, h in enumerate(hdrs2, 1): _hdr(ws2, 2, c, h, bg="1F618D")
    row = 3
    stages_order = ["R32","QF","SF","FINAL A","FINAL B","3o Lugar A","3o Lugar B"]
    for stage in stages_order:
        gms = [gm for gm in det["games"] if gm["stage"] == stage]
        if not gms: continue
        _hdr(ws2, row, 1, stage, bg="BDC3C7", fg="1A1A1A", bold=True)
        ws2.merge_cells(f"A{row}:J{row}"); row += 1
        for gm in gms:
            bg = STAGE_BG.get(stage, "FDFEFE")
            tipo = "✅ REAL" if gm["is_real"] else "🔮 Previsão"
            bg_tipo = "D5F5E3" if gm["is_real"] else bg
            vals = [gm["stage"], gm["mid"], tipo,
                    gm["t1"], gm["l1"], gm["score"], gm["l2"], gm["t2"],
                    gm["pk"], gm["winner"]]
            fmts = [None,None,None,None,"0.000",None,"0.000",None,"0.0%",None]
            bolds = [False,False,False,
                     gm["winner"]==gm["t1"],False,True,False,
                     gm["winner"]==gm["t2"],False,True]
            for c, (v, fmt, bold) in enumerate(zip(vals, fmts, bolds), 1):
                bg_c = "D4EFDF" if c == 10 else (bg_tipo if c == 3 else bg)
                _cell(ws2, row, c, v, bg=bg_c, bold=bold, fmt=fmt,
                      align="left" if c in (3,4,8,10) else "center")
            row += 1
    col_w2 = [8,4,12,26,7,10,7,26,10,26]
    for i, w in enumerate(col_w2, 1): ws2.column_dimensions[get_column_letter(i)].width = w

    # ── Aba 3: MC Probabilidades ──────────────────────────────────────────────
    ws3 = wb.create_sheet("MC_Probabilidades"); ws3.sheet_view.showGridLines = False
    _hdr(ws3, 1, 1, f"Monte Carlo {N_SIM:,} simulações — P(avançar) por fase", bg="154360")
    ws3.merge_cells("A1:I1")
    hdrs3 = ["Conf","Time","Em campo?","P(QF)","P(SF)","P(Final)","P(Campeão)","λ_atq","λ_def"]
    for c, h in enumerate(hdrs3, 1): _hdr(ws3, 2, c, h, bg="1F618D")
    row = 3
    still_in = set(KNOWN_R32_WINNERS.values()) | {
        t2 for mid, t1, t2, g1, g2, pen in R32_FULL if g1 is None
    } | {
        t1 for mid, t1, t2, g1, g2, pen in R32_FULL if g1 is None
    }
    sorted_teams = sorted(ALL_COPA_TEAMS,
                          key=lambda t: radv.get(t,{}).get("Campeon",0), reverse=True)
    for t in sorted_teams:
        conf   = COPA_TEAM_CONF.get(t, "?")
        ra     = radv.get(t, {})
        in_r32 = any(t in (t1, t2) for _, t1, t2, *_ in R32_FULL)
        # Teams with known real result
        eliminated = any(
            (mid in KNOWN_R32_WINNERS and KNOWN_R32_WINNERS[mid] != t and t in (t1, t2))
            for mid, t1, t2, *_ in R32_FULL
        )
        status = "❌ Eliminado" if eliminated else ("🟢 Em campo" if in_r32 else "–")
        lam_a  = np.mean([lam.get((t, o), 1.3) for o in ALL_COPA_TEAMS if o != t])
        lam_d  = np.mean([lam.get((o, t), 1.3) for o in ALL_COPA_TEAMS if o != t])
        pcamp  = ra.get("Campeon", 0) / N_SIM
        bg = ("FFDDD5" if pcamp>=0.10 else ("FFF3CD" if pcamp>=0.05 else
              ("D5F5E3" if pcamp>=0.02 else ("EBF5FB" if row%2==0 else "FDFEFE"))))
        if eliminated: bg = "F5F5F5"
        vals  = [conf, t, status,
                 ra.get("QF",0)/N_SIM, ra.get("SF",0)/N_SIM,
                 ra.get("Final",0)/N_SIM, pcamp,
                 round(lam_a,3), round(lam_d,3)]
        fmts  = [None,None,None,"0.0%","0.0%","0.0%","0.0%","0.000","0.000"]
        for c, (v, fmt) in enumerate(zip(vals, fmts), 1):
            _cell(ws3, row, c, v, bg=bg, bold=(pcamp>=0.05), fmt=fmt,
                  align="left" if c in (2,3) else "center")
        row += 1
    col_w3 = [10,26,14,9,9,9,12,9,9]
    for i, w in enumerate(col_w3, 1): ws3.column_dimensions[get_column_letter(i)].width = w

    # ── Aba 4: MC Campeão ─────────────────────────────────────────────────────
    ws4 = wb.create_sheet("MC_Campeon"); ws4.sheet_view.showGridLines = False
    _hdr(ws4, 1, 1, f"MC {N_SIM:,} sim — Campeões Copa 2026", bg="154360")
    ws4.merge_cells("A1:E1")
    for c, h in enumerate(["Rank","Conf","Time","Simulações","Probabilidade"], 1):
        _hdr(ws4, 2, c, h, bg="1F618D")
    row = 3
    fills = ["FFD700","C0C0C0","CD7F32"]
    total = sum(mc_champ.values())
    for rank, (team, cnt) in enumerate(
        sorted(mc_champ.items(), key=lambda x: x[1], reverse=True), 1
    ):
        pct = cnt / total
        conf = COPA_TEAM_CONF.get(team, "?")
        bg = (fills[rank-1] if rank <= 3 else
              ("FFDDD5" if pct>=0.10 else ("FFF3CD" if pct>=0.05 else
              ("D5F5E3" if pct>=0.02 else ("EBF5FB" if row%2==0 else "FDFEFE")))))
        for c, (v, fmt) in enumerate(
            zip([rank, conf, team, cnt, pct], [None,None,None,"0","0.0%"]), 1
        ):
            _cell(ws4, row, c, v, bg=bg, bold=(rank<=3), fmt=fmt,
                  align="left" if c==3 else "center")
        row += 1
    for i, w in enumerate([6,10,26,12,14], 1):
        ws4.column_dimensions[get_column_letter(i)].width = w

    wb.save(out)
    print(f"  Excel salvo: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{SEP}")
    print("  COPA 2026 v14 — R32 resultados reais + simulação restante")
    print(f"{SEP}\n")

    # 1. Rebuild model_dataset com CUTOFF=2026-07-03
    print("1. Reconstruindo model_dataset.csv (CUTOFF=2026-07-03)...")
    from build_features import (load_and_merge_raw, add_decay_features,
                                 impute_by_confederation, build_match_dataset,
                                 XI_DEFAULT, save_data)
    df_raw  = load_and_merge_raw()
    df_dec  = add_decay_features(df_raw, XI_DEFAULT)
    df_imp  = impute_by_confederation(df_dec)
    df_wide = build_match_dataset(df_imp)
    save_data(df_wide)
    wide    = pd.read_csv(MODEL_CSV, parse_dates=["date"])
    wide    = apply_saves_norm(wide)
    print(f"   model_dataset: {len(wide):,} jogos wide")
    wc_wide = wide[wide["match_type"] == "WC"]
    print(f"   WC no wide: {len(wc_wide)} jogos ({len(wc_wide[(wc_wide['date'] >= pd.Timestamp('2026-06-28'))])} inclui KO)")

    # 2. Long + winsorize + treino
    print("\n2. Treinando xgboost_v14...")
    long    = wide_to_long(wide)
    long_w  = winsorize(long)
    model   = train_model(long_w, CUTOFF)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"   Salvo: {OUT_PKL}")

    # 3. Lambdas
    print("\n3. Computando lambdas...")
    tf  = extract_feats(long_w)
    lam = compute_lambdas(model, tf, ALL_COPA_TEAMS)
    top5_atk = sorted(
        [(t, np.mean([lam.get((t,o),1.3) for o in ALL_COPA_TEAMS if o!=t]))
         for t in ALL_COPA_TEAMS], key=lambda x: x[1], reverse=True
    )[:5]
    print(f"   Top-5 λ_atk: {[(t, round(v,3)) for t,v in top5_atk]}")

    # 4. Standings
    print("\n4. Standings reais Copa 2026...")
    raw = pd.read_csv(RAW_CSV)
    standings = compute_standings(raw)
    for g in sorted(GROUPS.keys()):
        df = standings[g]
        print(f"   Grupo {g}: {df.iloc[0]['team']} ({df.iloc[0]['pts']}pts) | "
              f"{df.iloc[1]['team']} ({df.iloc[1]['pts']}pts)")

    # 5. Bracket determinístico
    print("\n5. Simulação determinística R32→Final...")
    det = simulate_deterministic(lam)

    print("\n   R32:")
    for gm in [g for g in det["games"] if g["stage"] == "R32"]:
        tag = "✅" if gm["is_real"] else "🔮"
        print(f"   {tag} M{gm['mid']:>3}  {gm['t1']:<25} {gm['score']:>12}  "
              f"{gm['t2']:<25}  → {gm['winner']}")

    print("\n   QF:")
    for gm in [g for g in det["games"] if g["stage"] == "QF"]:
        print(f"   🔮 M{gm['mid']:>3}  {gm['t1']:<25} {gm['score']:>12}  "
              f"{gm['t2']:<25}  → {gm['winner']}")

    print("\n   SF:")
    for gm in [g for g in det["games"] if g["stage"] == "SF"]:
        print(f"   🔮 M{gm['mid']:>3}  {gm['t1']:<25} {gm['score']:>12}  "
              f"{gm['t2']:<25}  → {gm['winner']}")

    print("\n   FINAIS:")
    for gm in [g for g in det["games"] if "FINAL" in g["stage"]]:
        print(f"   🔮 {gm['stage']}  {gm['t1']:<25} {gm['score']:>12}  "
              f"{gm['t2']:<25}  → {gm['winner']}")

    # 6. MC
    print(f"\n6. Monte Carlo {N_SIM:,} simulações...")
    mc_champ, radv = run_mc(lam)
    total = sum(mc_champ.values())
    print("   Top-10 campeões:")
    for rank, (team, cnt) in enumerate(
        sorted(mc_champ.items(), key=lambda x: x[1], reverse=True)[:10], 1
    ):
        print(f"   {rank:>2}. {team:<25} {cnt/total:5.1%}  (n={cnt})")

    # 7. Excel
    print("\n7. Gerando Excel...")
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    build_excel(OUT_XLSX, standings, det, mc_champ, radv, lam)

    print(f"\n{SEP}")
    print(f"  CONCLUÍDO")
    print(f"  Modelo:  {OUT_PKL}")
    print(f"  Excel:   {OUT_XLSX}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
