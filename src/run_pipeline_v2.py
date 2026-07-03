"""
run_pipeline_v2.py — Orquestração completa da pipeline enriquecida (Copa 2026 v2).

Sequência:
  1. fetch_friendlies_jun2026.py  — amistosos pré-Copa (mai-jun 2026, weight=0.8)
  2. fetch_player_ratings.py      — ratings de jogadores por jogo
  3. integrate_datasets.py        — une WCQ + extras + amistosos + jun2026 → full_dataset_raw.csv
  4. build_features.py            — rolling features + ratings → model_dataset.csv
  5. model_xgboost.py             — CV XGBoost vs Poisson (cutoff 2026-03-31) + SHAP
  6. simulate_copa2026.py         — 10.000 simulações MC → copa2026_monte_carlo.csv

Anti-leakage: amistosos jun/2026 entram no histórico dos times mas nunca como
target de predição. O CV usa apenas jogos até 2026-03-31.
"""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PYTHON = sys.executable

STEPS = [
    ("fetch_friendlies_jun2026",  "src/fetch_friendlies_jun2026.py"),
    ("fetch_player_ratings",      "src/fetch_player_ratings.py"),
    ("integrate_datasets",        "src/integrate_datasets.py"),
    ("build_features",            "src/build_features.py"),
    ("model_xgboost",             "src/model_xgboost.py"),
    ("simulate_copa2026",         "src/simulate_copa2026.py"),
]

OLD_CV_CSV   = Path("outputs/xgboost_vs_poisson.csv")
NEW_CV_CSV   = Path("outputs/xgboost_vs_poisson.csv")   # overwritten in-place
OLD_MC_CSV   = Path("outputs/copa2026_monte_carlo.csv")   # simulação v4 (avg5)
NEW_MC_CSV   = Path("outputs/copa2026_previsoes.csv")     # simulação v5 (decay)
FEATURES_CSV = Path("data/processed/model_dataset.csv")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _sep(title: str, width: int = 65) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print("=" * width)


def _run(label: str, script: str) -> bool:
    _sep(f"ETAPA — {label}")
    t0 = time.time()
    result = subprocess.run([PYTHON, script], capture_output=False)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"ERRO (returncode={result.returncode})"
    print(f"\n  [{label}] {status}  ({elapsed:.1f}s)")
    return ok


# ---------------------------------------------------------------------------
# Snapshot de métricas antigas (antes de sobrescrever)
# ---------------------------------------------------------------------------

def _snapshot_old_cv() -> dict | None:
    if not OLD_CV_CSV.exists():
        return None
    try:
        df = pd.read_csv(OLD_CV_CSV)
        # Espera colunas: fold, mae_xgb, mae_poisson, acc_xgb, acc_poisson, acc_away_xgb, acc_away_poisson
        agg = df.mean(numeric_only=True)
        return agg.to_dict()
    except Exception:
        return None


def _snapshot_old_mc() -> dict | None:
    if not OLD_MC_CSV.exists():
        return None
    try:
        df = pd.read_csv(OLD_MC_CSV)
        if "champion" in df.columns and "team" in df.columns:
            return dict(zip(df["team"], df["champion"]))
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Comparação de métricas CV
# ---------------------------------------------------------------------------

def _compare_cv(old: dict | None, label_old: str = "v1") -> None:
    if not NEW_CV_CSV.exists():
        print("  (arquivo CV não encontrado — pulando comparação)")
        return

    new_df = pd.read_csv(NEW_CV_CSV)
    new = new_df.mean(numeric_only=True).to_dict()

    _sep("COMPARAÇÃO CV (XGBoost) — antes e depois das novas features")

    keys = [
        ("mae_xgb",        "MAE XGBoost"),
        ("mae_poisson",     "MAE Poisson"),
        ("acc_xgb",         "Acc W/D/L XGBoost (%)"),
        ("acc_poisson",     "Acc W/D/L Poisson (%)"),
        ("acc_away_xgb",    "Acc Away Win XGBoost (%)"),
        ("acc_away_poisson","Acc Away Win Poisson (%)"),
    ]

    print(f"  {'Métrica':30s} {'v1 (antes)':>12} {'v2 (depois)':>12} {'Delta':>10}")
    print("  " + "-" * 68)

    for key, label in keys:
        v_old = old.get(key, float("nan")) if old else float("nan")
        v_new = new.get(key, float("nan"))
        delta = v_new - v_old if (v_old == v_old and v_new == v_new) else float("nan")
        old_s = f"{v_old:.4f}" if v_old == v_old else "   n/a"
        new_s = f"{v_new:.4f}" if v_new == v_new else "   n/a"
        dlt_s = f"{delta:+.4f}" if delta == delta else "   n/a"
        print(f"  {label:30s} {old_s:>12} {new_s:>12} {dlt_s:>10}")


# ---------------------------------------------------------------------------
# Comparação de simulação MC
# ---------------------------------------------------------------------------

def _compare_mc(old_champ: dict | None) -> None:
    if not NEW_MC_CSV.exists():
        print("  (arquivo MC não encontrado)")
        return

    new_df = pd.read_csv(NEW_MC_CSV).sort_values("champion", ascending=False)

    _sep("NOVO TOP-10 CAMPEÕES MAIS PROVÁVEIS (v2)")
    print(f"  {'#':>3} {'Time':25s} {'v2 %':>8} {'v1 %':>8}  {'Delta':>7}")
    print("  " + "-" * 58)
    for rank, (_, row) in enumerate(new_df.head(10).iterrows(), 1):
        team = row["team"]
        v2   = row["champion"]
        v1   = old_champ.get(team, float("nan")) if old_champ else float("nan")
        dlt  = v2 - v1 if v1 == v1 else float("nan")
        v1_s = f"{v1:.1f}" if v1 == v1 else "  n/a"
        dlt_s = f"{dlt:+.1f}" if dlt == dlt else "  n/a"
        bar  = "█" * max(1, int(v2 / new_df["champion"].iloc[0] * 20))
        print(f"  {rank:3d} {team:25s} {v2:8.1f} {v1_s:>8} {dlt_s:>8}  {bar}")

    if old_champ:
        _sep("MUDANÇAS MAIS RELEVANTES (quem subiu / quem caiu)")
        changes = []
        for _, row in new_df.iterrows():
            team = row["team"]
            v2   = row["champion"]
            v1   = old_champ.get(team, 0.0)
            changes.append((team, v1, v2, v2 - v1))

        changes.sort(key=lambda x: abs(x[3]), reverse=True)
        print(f"  {'Time':25s} {'v1 %':>8} {'v2 %':>8} {'Delta':>8}")
        print("  " + "-" * 55)
        for team, v1, v2, delta in changes[:15]:
            direction = "↑" if delta > 0 else "↓"
            print(f"  {team:25s} {v1:8.1f} {v2:8.1f} {delta:+8.1f} {direction}")


# ---------------------------------------------------------------------------
# Cobertura de rating_medio_avg5
# ---------------------------------------------------------------------------

COPA2026_TEAMS = [
    "Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay", "Paraguay",
    "Germany", "France", "Spain", "England", "Netherlands", "Portugal",
    "Belgium", "Austria", "Switzerland", "Croatia",
    "Norway", "Sweden", "Czech Republic", "Turkey",
    "Scotland", "Bosnia and Herzegovina",
    "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia",
    "Uzbekistan", "Jordan", "Iraq", "Qatar",
    "Morocco", "Senegal", "Egypt", "Ghana", "Cape Verde",
    "DR Congo", "Ivory Coast", "South Africa", "Algeria", "Tunisia",
    "United States", "Mexico", "Canada", "Panama", "Curacao", "Haiti",
    "New Zealand",
]


def _rating_coverage() -> None:
    if not FEATURES_CSV.exists():
        print("  model_dataset.csv não encontrado — pulando análise de cobertura.")
        return

    df = pd.read_csv(FEATURES_CSV)

    # Colunas de decay no wide format (v5)
    decay_col = None
    for candidate in ["home_gls_decay", "away_gls_decay"]:
        if candidate in df.columns:
            decay_col = candidate
            break

    if decay_col is None:
        print("  Colunas _decay não encontradas em model_dataset.csv.")
        return

    # Pegar times com features preenchidas
    teams_covered = set()
    for side in ("home", "away"):
        col      = f"{side}_gls_decay"
        team_col = f"{side}_team"
        if col not in df.columns or team_col not in df.columns:
            continue
        sub = df[df[col].notna()][[team_col]].copy()
        teams_covered.update(sub[team_col].tolist())

    copa_with    = [t for t in COPA2026_TEAMS if t in teams_covered]
    copa_without = [t for t in COPA2026_TEAMS if t not in teams_covered]

    _sep("COBERTURA DE gls_decay NOS 48 TIMES (v5)")
    print(f"  Times Copa 2026 com gls_decay: {len(copa_with)} / {len(COPA2026_TEAMS)}")
    if copa_without:
        print(f"\n  Sem dados de decay (imputação por confederação):")
        for t in sorted(copa_without):
            print(f"    {t}")
    else:
        print("  Todos os 48 times têm cobertura de decay!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _sep("run_pipeline_v2.py — Copa 2026 Pipeline Enriquecida", width=65)
    print("  Cutoff CV: 2026-03-31  |  Jun/2026 = só features, nunca target")
    print("  Novas features: rating_medio_avg5 + condicionais forte/fraco")

    # 1. Snapshot do estado ANTES de rodar (para comparação posterior)
    old_cv_metrics  = _snapshot_old_cv()
    old_mc_champion = _snapshot_old_mc()

    # OLD_MC_CSV = copa2026_monte_carlo.csv (v4/avg5) é preservado sem sobrescrever

    # 2. Executa cada etapa em sequência
    failed_at = None
    for label, script in STEPS:
        ok = _run(label, script)
        if not ok:
            failed_at = label
            break

    if failed_at:
        _sep(f"PIPELINE INTERROMPIDA EM: {failed_at}")
        print(f"  Verifique os erros acima e corrija antes de continuar.")
        sys.exit(1)

    # 3. Relatório final
    _sep("RELATÓRIO FINAL — Pipeline v2", width=65)
    _compare_cv(old_cv_metrics)
    _compare_mc(old_mc_champion)
    _rating_coverage()

    _sep("PIPELINE CONCLUÍDA", width=65)
    print(f"  Outputs:")
    print(f"    outputs/xgboost_v5.pkl            — modelo decay v5")
    print(f"    outputs/xgboost_vs_poisson.csv    — métricas CV")
    print(f"    outputs/shap_importance.png        — SHAP features")
    print(f"    outputs/copa2026_bracket.txt       — bracket determinístico")
    print(f"    outputs/copa2026_previsoes.csv     — MC v5 com todos os 48 times")
    if OLD_MC_CSV.exists():
        print(f"    outputs/copa2026_monte_carlo.csv   — simulação v4 (avg5) preservada")


if __name__ == "__main__":
    main()
