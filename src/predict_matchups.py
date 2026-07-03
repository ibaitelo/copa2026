"""
predict_matchups.py — Previsão de placares para jogos específicos da Copa 2026.

Usa o modelo v4 treinado (outputs/xgboost_v4.pkl) + features recentes de cada seleção.
Para cada jogo mostra:
  - λ esperado (gols esperados por time)
  - Placar mais provável
  - Top-5 placares mais prováveis
  - P(vitória t1) / P(empate) / P(vitória t2)
"""

import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as sci_poisson

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_PKL   = Path("outputs/xgboost_v4.pkl")
INPUT_CSV = Path("data/processed/model_dataset.csv")
ELO_CSV   = Path("data/raw/elo_history.csv")

# ─── Features (idêntico ao simulate_copa2026.py) ─────────────────────────────
BASE_FEATURES = [
    "gls_avg5", "shots_on_goal_avg5", "win_rate_avg5", "opp_saves_avg5",
    "home_advantage",
]
RATING_FEATURES = [
    "delta_rating_avg5",
    "gls_avg5_vs_forte", "gls_avg5_vs_fraco", "shots_avg5_vs_forte",
    "gls_ponderado_avg5",
    "elo_diff_avg5",
    "rating_titulares_avg5",
]
CONF_DUMMIES = [
    "conf_AFC", "conf_CAF", "conf_CONCACAF", "conf_CONMEBOL",
    "conf_FRIENDLY", "conf_OFC", "conf_UEFA",
]
FEATURES = BASE_FEATURES + RATING_FEATURES + CONF_DUMMIES + ["match_type_wcq"]

# ─── Grupos Copa 2026 ─────────────────────────────────────────────────────────
GROUPS = {
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


def load_team_features() -> dict[str, dict]:
    """Reconstrói team_feats usando o mesmo código do simulate_copa2026."""
    wide = pd.read_csv(INPUT_CSV, parse_dates=["date"])

    rows = []
    for _, r in wide.iterrows():
        for side, opp in [("home", "away"), ("away", "home")]:
            row = {
                "date":  r["date"],
                "team":  r[f"{side}_team"],
                "side":  side,
                "gls_avg5":           r.get(f"{side}_gls_avg5", np.nan),
                "shots_on_goal_avg5": r.get(f"{side}_shots_on_goal_avg5", np.nan),
                "win_rate_avg5":      r.get(f"{side}_win_rate_avg5", np.nan),
                "saves_avg5":         r.get(f"{side}_saves_avg5", np.nan),
                "opp_saves_avg5":     r.get(f"{opp}_saves_avg5", np.nan),
            }
            for col in RATING_FEATURES:
                row[col] = r.get(f"{side}_{col}", np.nan)
            rows.append(row)

    long = pd.DataFrame(rows).sort_values("date")

    team_feats: dict[str, dict] = {}
    for _, row in long.iterrows():
        entry = {
            "gls_avg5":           float(row["gls_avg5"]) if pd.notna(row["gls_avg5"]) else 1.2,
            "shots_on_goal_avg5": float(row["shots_on_goal_avg5"]) if pd.notna(row["shots_on_goal_avg5"]) else 3.5,
            "win_rate_avg5":      float(row["win_rate_avg5"]) if pd.notna(row["win_rate_avg5"]) else 0.4,
            "saves_avg5":         float(row["saves_avg5"]) if pd.notna(row["saves_avg5"]) else 1.8,
        }
        for col in RATING_FEATURES:
            v = row.get(col, np.nan)
            entry[col] = float(v) if pd.notna(v) else np.nan
        team_feats[row["team"]] = entry

    # ELO atual
    if ELO_CSV.exists():
        elo_hist = pd.read_csv(ELO_CSV, parse_dates=["date"])
        current_elo = (elo_hist.sort_values("date")
                       .groupby("team")["elo_after"].last()
                       .to_dict())
        for team, elo_val in current_elo.items():
            if team in team_feats:
                team_feats[team]["_elo_current"] = float(elo_val)
            else:
                team_feats[team] = {"_elo_current": float(elo_val)}

    return team_feats


def build_row(t1: str, t2: str, team_feats: dict) -> dict:
    """Monta uma linha de features para o par t1 vs t2 (campo neutro)."""
    f1 = team_feats.get(t1, {})
    f2 = team_feats.get(t2, {})
    row = {
        "gls_avg5":           f1.get("gls_avg5", 1.2),
        "shots_on_goal_avg5": f1.get("shots_on_goal_avg5", 3.5),
        "win_rate_avg5":      f1.get("win_rate_avg5", 0.4),
        "opp_saves_avg5":     f2.get("saves_avg5", 2.0),
        "home_advantage":     0,
        "match_type_wcq":     0,
        "conf_AFC": 0, "conf_CAF": 0, "conf_CONCACAF": 0,
        "conf_CONMEBOL": 0, "conf_FRIENDLY": 0, "conf_OFC": 0, "conf_UEFA": 0,
    }
    for col in RATING_FEATURES:
        if col == "elo_diff_avg5":
            e1 = f1.get("_elo_current", np.nan)
            e2 = f2.get("_elo_current", np.nan)
            if pd.notna(e1) and pd.notna(e2):
                row[col] = float(e1) - float(e2)
            else:
                row[col] = f1.get(col, np.nan)
        else:
            row[col] = f1.get(col, np.nan)
    return row


def get_lambda(model, t1: str, t2: str, team_feats: dict) -> float:
    row = build_row(t1, t2, team_feats)
    df = pd.DataFrame([row])
    return float(model.predict(df[FEATURES])[0])


def score_probs(lam1: float, lam2: float, max_g: int = 8):
    """Retorna dict {(g1, g2): probabilidade}."""
    probs = {}
    for g1 in range(max_g + 1):
        for g2 in range(max_g + 1):
            probs[(g1, g2)] = sci_poisson.pmf(g1, lam1) * sci_poisson.pmf(g2, lam2)
    return probs


def analyze_matchup(model, t1: str, t2: str, team_feats: dict) -> None:
    lam1 = get_lambda(model, t1, t2, team_feats)
    lam2 = get_lambda(model, t2, t1, team_feats)
    probs = score_probs(lam1, lam2)

    p_win  = sum(p for (g1, g2), p in probs.items() if g1 > g2)
    p_draw = sum(p for (g1, g2), p in probs.items() if g1 == g2)
    p_loss = sum(p for (g1, g2), p in probs.items() if g1 < g2)

    top5 = sorted(probs.items(), key=lambda x: -x[1])[:5]

    elo1 = team_feats.get(t1, {}).get("_elo_current", float("nan"))
    elo2 = team_feats.get(t2, {}).get("_elo_current", float("nan"))
    elo_str = f"ELO {elo1:.0f} × {elo2:.0f}" if (not np.isnan(elo1) and not np.isnan(elo2)) else ""

    print(f"\n  {t1:22s} vs  {t2}")
    print(f"  {'─'*48}")
    print(f"  λ esperado:   {lam1:.3f}  ×  {lam2:.3f}      {elo_str}")
    print(f"  Resultado:    {p_win*100:.1f}% V | {p_draw*100:.1f}% E | {p_loss*100:.1f}% D")
    print(f"  Top placares:")
    for (g1, g2), p in top5:
        bar = "█" * int(p * 200)
        arrow = "←" if g1 > g2 else ("→" if g1 < g2 else "═")
        print(f"    {g1}–{g2}  {arrow}  {p*100:.1f}%  {bar}")


def print_group(model, group_name: str, teams: list, team_feats: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  GRUPO {group_name}  —  {' / '.join(teams)}")
    print(f"{'='*55}")
    from itertools import combinations
    for t1, t2 in combinations(teams, 2):
        analyze_matchup(model, t1, t2, team_feats)


def main() -> None:
    print("=" * 55)
    print("  PREVISÃO DE PLACARES — COPA 2026  (modelo v4)")
    print("=" * 55)

    if not OUT_PKL.exists():
        print(f"ERRO: {OUT_PKL} não encontrado. Execute simulate_copa2026.py primeiro.")
        sys.exit(1)

    with open(OUT_PKL, "rb") as f:
        model = pickle.load(f)

    print("  Modelo v4 carregado.")
    team_feats = load_team_features()
    print(f"  Features de {len(team_feats)} seleções carregadas.\n")

    # ── Grupos de foco ────────────────────────────────────────────────────────
    focus_groups = ["C", "J", "I", "E", "H", "K"]  # Brasil, Argentina, França, etc.

    for g in focus_groups:
        print_group(model, g, GROUPS[g], team_feats)

    # ── Confrontos específicos adicionais ────────────────────────────────────
    print(f"\n{'='*55}")
    print("  CONFRONTOS ESPECIAIS (possíveis mata-matas)")
    print(f"{'='*55}")
    specials = [
        ("Brazil",    "Argentina"),
        ("Brazil",    "France"),
        ("Brazil",    "Germany"),
        ("Brazil",    "Spain"),
        ("Argentina", "France"),
        ("Argentina", "Germany"),
        ("France",    "Germany"),
        ("Spain",     "Germany"),
        ("Morocco",   "France"),
        ("England",   "Germany"),
    ]
    for t1, t2 in specials:
        analyze_matchup(model, t1, t2, team_feats)

    print()


if __name__ == "__main__":
    main()
