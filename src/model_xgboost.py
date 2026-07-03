"""
model_xgboost.py — XGBoost count:poisson vs GLM Poisson
Features v10: 16 features (opp_saves_decay restaurado após Panama inflacionar sem ele).

Saídas:
  outputs/xgboost_v10.pkl          — modelo treinado v10
  outputs/xgboost_vs_poisson.csv   — métricas por fold + global
  outputs/shap_importance_v10.png  — top-10 features SHAP v10
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap
import statsmodels.api as sm
from scipy.stats import poisson
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# make build_features importable from the same src/ directory
sys.path.insert(0, str(Path(__file__).parent))

INPUT_CSV  = Path("data/processed/model_dataset.csv")
OUT_CSV    = Path("outputs/xgboost_vs_poisson.csv")
OUT_SHAP   = Path("outputs/shap_importance_v10.png")
OUT_PKL    = Path("outputs/xgboost_v10.pkl")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parâmetros de CV
# ---------------------------------------------------------------------------
CUTOFF    = "2026-03-31"
MIN_TRAIN = 30
FOLD_SIZE = 5

# ξ a testar na otimização
XI_VALUES   = [0.001, 0.002, 0.003, 0.004, 0.005]
XI_DEFAULT  = 0.003  # fallback se build_features indisponível

# ---------------------------------------------------------------------------
# Features v10 — 16 features (opp_saves_decay restaurado após Panama inflacionar 1.1%→6.7%)
# ---------------------------------------------------------------------------
CONF_DUMMIES = ["conf_AFC", "conf_CAF", "conf_CONCACAF", "conf_CONMEBOL",
                "conf_OFC", "conf_UEFA"]

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
FEATURES = BASE_FEATURES + RATING_FEATURES + CONF_DUMMIES  # 16 features

POISSON_FEATURES = [
    "opp_saves_decay",
    "shots_on_goal_decay",
    "gols_sofridos_decay",
    "host_factor",
]

VERSION_HISTORY = {
    "v2 (c/ratings)":          {"mae_xgb": 1.094, "acc_xgb": 0.459},
    "v4 (ELO+titulares)":      {"mae_xgb": 0.899, "acc_xgb": 0.464},
    "v7 (fix saves_0)":        {"mae_xgb": 0.891, "acc_xgb": 0.465},
    "v8 (feature redesign)":   {"mae_xgb": 0.890, "acc_xgb": 0.423},
    "v9 (ELO-based gls_pond)": {"mae_xgb": 0.8751, "acc_xgb": 0.308},
}

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


# ---------------------------------------------------------------------------
# Wide → Long
# ---------------------------------------------------------------------------

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for side, opp in [("home", "away"), ("away", "home")]:
            # host_factor em treino: reutiliza home_advantage (0/1 WCQ/friendlies)
            hf = int(r.get("home_advantage", 0)) if side == "home" else 0
            row = {
                "match_id":            r["match_id"],
                "date":                r["date"],
                "home_team":           r["home_team"],
                "away_team":           r["away_team"],
                "team":                r[f"{side}_team"],
                "opponent":            r[f"{opp}_team"],
                "side":                side,
                "gols_marcados":       r[f"{side}_gols"],
                "opp_saves_decay":     float(r.get(f"{opp}_saves_decay", 2.0)),
                "shots_on_goal_decay": r[f"{side}_shots_on_goal_decay"],
                "gols_sofridos_decay": r.get(f"{side}_gols_sofridos_decay", np.nan),
                "host_factor":         hf,
            }
            for col in CONF_DUMMIES:
                row[col] = int(r.get(col, 0))
            for col in RATING_FEATURES:
                raw = r.get(f"{side}_{col}", np.nan)
                row[col] = float(raw) if pd.notna(raw) else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Score matrix
# ---------------------------------------------------------------------------

def most_likely_score(lh: float, la: float, max_g: int = 8) -> tuple[int, int]:
    best_p, bh, ba = -1.0, 0, 0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = poisson.pmf(h, max(lh, 1e-9)) * poisson.pmf(a, max(la, 1e-9))
            if p > best_p:
                best_p, bh, ba = p, h, a
    return bh, ba


def result_label(h: int, a: int) -> str:
    return "H" if h > a else ("A" if h < a else "D")


# ---------------------------------------------------------------------------
# Expanding-window CV
# ---------------------------------------------------------------------------

def get_folds(wide: pd.DataFrame):
    wide_sorted = wide.sort_values("date").reset_index(drop=True)
    n = len(wide_sorted)
    folds = []
    train_end = MIN_TRAIN
    while train_end < n:
        test_end = min(train_end + FOLD_SIZE, n)
        folds.append((list(range(train_end)), list(range(train_end, test_end))))
        train_end = test_end
    return wide_sorted, folds


def fit_poisson_model(train_long: pd.DataFrame):
    X = sm.add_constant(train_long[POISSON_FEATURES], has_constant="add")
    y = train_long["gols_marcados"].astype(float)
    return sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.Log())).fit(disp=False)


def predict_poisson(result, df_long: pd.DataFrame) -> np.ndarray:
    X = sm.add_constant(df_long[POISSON_FEATURES], has_constant="add")
    return result.predict(X).to_numpy()


def fit_xgb_model(train_long: pd.DataFrame) -> XGBRegressor:
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train_long[FEATURES], train_long["gols_marcados"])
    return model


def predict_xgb(model: XGBRegressor, df_long: pd.DataFrame) -> np.ndarray:
    return model.predict(df_long[FEATURES])


def evaluate_games(pred_records: list[dict]) -> dict:
    if not pred_records:
        return {}
    df = pd.DataFrame(pred_records)
    errors_h = np.abs(df["home_lambda"] - df["home_gols"])
    errors_a = np.abs(df["away_lambda"] - df["away_gols"])
    all_errors = pd.concat([errors_h, errors_a])

    df["result_real"] = df.apply(lambda r: result_label(r["home_gols"], r["away_gols"]), axis=1)
    df["result_pred"] = df.apply(lambda r: result_label(r["pred_home"], r["pred_away"]), axis=1)
    df["correct"]     = df["result_real"] == df["result_pred"]

    acc = {}
    for res in ["H", "D", "A"]:
        mask = df["result_real"] == res
        acc[f"acc_{res}"] = df.loc[mask, "correct"].mean() if mask.any() else float("nan")
        acc[f"n_{res}"]   = int(mask.sum())

    return {
        "n_games":  len(df),
        "mae":      float(all_errors.mean()),
        "rmse":     float(np.sqrt((all_errors ** 2).mean())),
        "acc_wdl":  float(df["correct"].mean()),
        **acc,
    }


def score_dist(pred_records: list[dict]) -> dict:
    if not pred_records:
        return {}
    df = pd.DataFrame(pred_records)
    real = df.apply(lambda r: f"{int(r['home_gols'])}-{int(r['away_gols'])}", axis=1)
    pred = df.apply(lambda r: f"{int(r['pred_home'])}-{int(r['pred_away'])}", axis=1)
    return {"real_top5": real.value_counts().head(5).to_dict(),
            "pred_top5": pred.value_counts().head(5).to_dict()}


def run_cv(wide: pd.DataFrame, long: pd.DataFrame, verbose: bool = True):
    wide_sorted, folds = get_folds(wide)

    fold_results     = []
    all_preds_xgb    = []
    all_preds_poisson = []

    if verbose:
        print(f"  Executando CV: {len(folds)} folds, MIN_TRAIN={MIN_TRAIN}, FOLD_SIZE={FOLD_SIZE}")
        print(f"  {'Fold':>5}  {'Treino':>7}  {'Teste':>6}  "
              f"{'MAE-XGB':>9}  {'MAE-Poi':>9}  {'AccXGB':>7}  {'AccPoi':>7}")
        print("  " + "-" * 65)

    for fold_i, (train_wide_idx, test_wide_idx) in enumerate(folds):
        train_games = wide_sorted.iloc[train_wide_idx]
        test_games  = wide_sorted.iloc[test_wide_idx]

        train_mids = set(train_games["match_id"])
        test_mids  = set(test_games["match_id"])
        train_long = long[long["match_id"].isin(train_mids)].copy()
        test_long  = long[long["match_id"].isin(test_mids)].copy()

        if train_long.empty or test_long.empty:
            continue

        xgb_model = fit_xgb_model(train_long)
        try:
            poi_model = fit_poisson_model(train_long)
        except Exception:
            poi_model = None

        test_long["lambda_xgb"] = predict_xgb(xgb_model, test_long)
        if poi_model is not None:
            test_long["lambda_poi"] = predict_poisson(poi_model, test_long)
        else:
            test_long["lambda_poi"] = test_long["gols_marcados"].mean()

        fold_preds_xgb = []
        fold_preds_poi = []

        for mid in test_mids:
            sub      = test_long[test_long["match_id"] == mid]
            home_row = sub[sub["side"] == "home"]
            away_row = sub[sub["side"] == "away"]
            if home_row.empty or away_row.empty:
                continue

            hg = int(home_row["gols_marcados"].iloc[0])
            ag = int(away_row["gols_marcados"].iloc[0])

            lh_x = float(home_row["lambda_xgb"].iloc[0])
            la_x = float(away_row["lambda_xgb"].iloc[0])
            ph_x, pa_x = most_likely_score(lh_x, la_x)

            lh_p = float(home_row["lambda_poi"].iloc[0])
            la_p = float(away_row["lambda_poi"].iloc[0])
            ph_p, pa_p = most_likely_score(lh_p, la_p)

            rec = dict(home_gols=hg, away_gols=ag,
                       home_team=home_row["home_team"].iloc[0],
                       away_team=home_row["away_team"].iloc[0])

            fold_preds_xgb.append({**rec, "home_lambda": lh_x, "away_lambda": la_x,
                                    "pred_home": ph_x, "pred_away": pa_x})
            fold_preds_poi.append({**rec, "home_lambda": lh_p, "away_lambda": la_p,
                                    "pred_home": ph_p, "pred_away": pa_p})

        all_preds_xgb.extend(fold_preds_xgb)
        all_preds_poisson.extend(fold_preds_poi)

        m_xgb = evaluate_games(fold_preds_xgb)
        m_poi = evaluate_games(fold_preds_poi)

        fold_results.append({
            "fold":        fold_i + 1,
            "n_train":     len(train_games),
            "n_test":      len(test_games),
            "mae_xgb":     m_xgb.get("mae", float("nan")),
            "mae_poisson": m_poi.get("mae", float("nan")),
            "acc_xgb":     m_xgb.get("acc_wdl", float("nan")),
            "acc_poisson": m_poi.get("acc_wdl", float("nan")),
        })

        if verbose:
            print(f"  {fold_i+1:5d}  {len(train_games):7d}  {len(test_games):6d}  "
                  f"{m_xgb.get('mae', float('nan')):9.4f}  "
                  f"{m_poi.get('mae', float('nan')):9.4f}  "
                  f"{m_xgb.get('acc_wdl', float('nan')):7.1%}  "
                  f"{m_poi.get('acc_wdl', float('nan')):7.1%}")

    return fold_results, all_preds_xgb, all_preds_poisson


# ---------------------------------------------------------------------------
# Otimização de ξ via expanding-window CV
# ---------------------------------------------------------------------------

def tune_xi(xi_values: list[float] = None) -> float:
    """
    Testa valores de ξ via expanding-window CV idêntico ao modelo final.
    Imprime tabela MAE por ξ e retorna o melhor (menor MAE médio).
    """
    if xi_values is None:
        xi_values = XI_VALUES

    try:
        import build_features as bf
    except ImportError:
        print("  [tune_xi] build_features não disponível — usando ξ padrão")
        return XI_DEFAULT

    sep = "=" * 62
    print(f"\n{sep}")
    print("  OTIMIZAÇÃO DE ξ — decay temporal exponencial")
    print(f"  Valores testados: {xi_values}")
    print(f"{sep}")

    print("\n  Carregando e mesclando dados brutos (uma vez para todos os ξ)...")
    df_raw = bf.load_and_merge_raw()

    results: list[dict] = []
    for xi in xi_values:
        print(f"\n  ── ξ = {xi:.3f} ──────────────────────────────")
        df_dec  = bf.add_decay_features(df_raw.copy(), xi, verbose=False)
        df_imp  = bf.impute_by_confederation(df_dec, verbose=False)
        df_wide = bf.build_match_dataset(df_imp)

        # dropna nas features core
        core_w = [c for c in df_wide.columns
                  if c.endswith("_decay") and not c.startswith("has_")]
        df_wide = df_wide.dropna(subset=core_w).reset_index(drop=True)

        df_long = wide_to_long(df_wide)
        print(f"  Dataset: {len(df_wide)} jogos | Long: {len(df_long)} linhas")

        _, cv_preds, _ = run_cv(df_wide, df_long, verbose=False)
        metrics = evaluate_games(cv_preds)
        mae = metrics.get("mae", float("nan"))
        acc = metrics.get("acc_wdl", float("nan"))
        print(f"  MAE_CV={mae:.4f}  ACC_CV={acc:.1%}")
        results.append({"xi": xi, "mae": mae, "acc": acc})

    print(f"\n{sep}")
    print(f"  TABELA RESUMO — ξ vs MAE_CV")
    print(f"  {'ξ':>8}  {'MAE_CV':>9}  {'ACC_CV':>8}")
    print(f"  {'-'*30}")
    best_mae = min(r["mae"] for r in results if not np.isnan(r["mae"]))
    for r in results:
        marker = "  ← melhor" if abs(r["mae"] - best_mae) < 1e-9 else ""
        print(f"  {r['xi']:>8.3f}  {r['mae']:>9.4f}  {r['acc']:>8.1%}{marker}")

    best_xi = min(results, key=lambda r: r["mae"] if not np.isnan(r["mae"]) else 1e9)["xi"]
    print(f"\n  → ξ ótimo selecionado: {best_xi:.3f}")
    print(f"{sep}\n")
    return best_xi


# ---------------------------------------------------------------------------
# Teste final (pós cutoff)
# ---------------------------------------------------------------------------

def run_final_test(wide: pd.DataFrame, long: pd.DataFrame):
    train_wide = wide[wide["date"] <= CUTOFF]
    test_wide  = wide[wide["date"] >  CUTOFF]
    train_long = long[long["match_id"].isin(train_wide["match_id"])]
    test_long  = long[long["match_id"].isin(test_wide["match_id"])].copy()

    print(f"\n  Treino final: {len(train_wide)} jogos | Teste: {len(test_wide)} jogos")

    xgb_model = fit_xgb_model(train_long)
    poi_model = fit_poisson_model(train_long)

    test_long["lambda_xgb"] = predict_xgb(xgb_model, test_long)
    test_long["lambda_poi"] = predict_poisson(poi_model, test_long)

    preds_xgb, preds_poi = [], []
    for mid in test_wide["match_id"].unique():
        sub      = test_long[test_long["match_id"] == mid]
        home_row = sub[sub["side"] == "home"]
        away_row = sub[sub["side"] == "away"]
        if home_row.empty or away_row.empty:
            continue

        hg = int(home_row["gols_marcados"].iloc[0])
        ag = int(away_row["gols_marcados"].iloc[0])
        rec = dict(home_gols=hg, away_gols=ag,
                   home_team=home_row["home_team"].iloc[0],
                   away_team=home_row["away_team"].iloc[0])

        lh_x, la_x = float(home_row["lambda_xgb"].iloc[0]), float(away_row["lambda_xgb"].iloc[0])
        ph_x, pa_x = most_likely_score(lh_x, la_x)
        preds_xgb.append({**rec, "home_lambda": lh_x, "away_lambda": la_x,
                           "pred_home": ph_x, "pred_away": pa_x})

        lh_p, la_p = float(home_row["lambda_poi"].iloc[0]), float(away_row["lambda_poi"].iloc[0])
        ph_p, pa_p = most_likely_score(lh_p, la_p)
        preds_poi.append({**rec, "home_lambda": lh_p, "away_lambda": la_p,
                          "pred_home": ph_p, "pred_away": pa_p})

    return xgb_model, train_long, preds_xgb, preds_poi


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_and_plot_shap(xgb_model: XGBRegressor, train_long: pd.DataFrame):
    sample    = train_long[FEATURES].sample(min(1000, len(train_long)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer(sample, check_additivity=False)

    mean_abs = pd.Series(
        np.abs(shap_vals.values).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)

    top10 = mean_abs.head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c" if f in BASE_FEATURES else "#3498db" for f in top10.index]
    ax.barh(range(len(top10)), top10.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10.index.tolist()[::-1], fontsize=10)
    ax.set_xlabel("Importância SHAP média (|valor|)", fontsize=10)
    ax.set_title("Top-10 Features — XGBoost v10 (16 features)\n(vermelho=base | azul=rating/contexto)",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_SHAP, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico SHAP salvo: {OUT_SHAP}")
    return mean_abs


# ---------------------------------------------------------------------------
# Tabela de distribuição de placares
# ---------------------------------------------------------------------------

def print_score_dist_table(preds_xgb, preds_poi):
    dist_xgb = score_dist(preds_xgb)
    dist_poi = score_dist(preds_poi)

    print("\n  Placares mais frequentes (conjunto de teste final):")
    print(f"  {'Placar':10s} {'Real':>7}  |  {'XGB':>7}  {'Poisson':>9}")
    print("  " + "-" * 42)

    real_counts = dist_xgb.get("real_top5", {})
    all_scores  = (set(real_counts) | set(dist_xgb.get("pred_top5", {}))
                   | set(dist_poi.get("pred_top5", {})))
    for sc in sorted(all_scores, key=lambda s: real_counts.get(s, 0), reverse=True)[:10]:
        r  = real_counts.get(sc, 0)
        xg = dist_xgb.get("pred_top5", {}).get(sc, 0)
        po = dist_poi.get("pred_top5", {}).get(sc, 0)
        print(f"  {sc:10s} {r:7d}  |  {xg:7d}  {po:9d}")

    def result_counts(preds, label):
        df = pd.DataFrame(preds)
        pred_res = df.apply(lambda r: result_label(r["pred_home"], r["pred_away"]), axis=1)
        real_res = df.apply(lambda r: result_label(r["home_gols"], r["away_gols"]), axis=1)
        print(f"\n  Distribuição de resultados previstos ({label}):")
        print(f"    Real    : H={int((real_res=='H').sum())}  D={int((real_res=='D').sum())}  A={int((real_res=='A').sum())}")
        print(f"    Previsto: H={int((pred_res=='H').sum())}  D={int((pred_res=='D').sum())}  A={int((pred_res=='A').sum())}")

    result_counts(preds_xgb, "XGBoost")
    result_counts(preds_poi, "Poisson")


# ---------------------------------------------------------------------------
# Salva CSV comparativo
# ---------------------------------------------------------------------------

def save_comparison_csv(fold_results, metrics_xgb, metrics_poi):
    rows = [{"tipo": "fold", **fr} for fr in fold_results]

    summary = {
        "tipo": "GLOBAL_TEST", "fold": "—", "n_train": "—",
        "n_test":     metrics_xgb.get("n_games"),
        "mae_xgb":    metrics_xgb.get("mae"),
        "mae_poisson": metrics_poi.get("mae"),
        "acc_xgb":    metrics_xgb.get("acc_wdl"),
        "acc_poisson": metrics_poi.get("acc_wdl"),
    }
    for res in ["H", "D", "A"]:
        summary[f"acc_{res}_xgb"]     = metrics_xgb.get(f"acc_{res}")
        summary[f"acc_{res}_poisson"] = metrics_poi.get(f"acc_{res}")
        summary[f"n_{res}_real"]      = metrics_xgb.get(f"n_{res}")

    df_out = pd.DataFrame(rows + [summary])
    target = OUT_CSV
    try:
        df_out.to_csv(target, index=False, encoding="utf-8")
    except PermissionError:
        target = target.with_stem(target.stem + "_v2")
        df_out.to_csv(target, index=False, encoding="utf-8")
    print(f"\n  Métricas salvas: {target}")


# ---------------------------------------------------------------------------
# Tabela comparativa final
# ---------------------------------------------------------------------------

def print_final_table(metrics_xgb: dict, metrics_poi: dict, shap_top: pd.Series,
                      best_xi: float = XI_DEFAULT):
    sep = "=" * 62

    def fmt(v, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  N/A  "
        return f"{v:.1%}" if pct else f"{v:.4f}"

    delta_mae = metrics_xgb["mae"] - metrics_poi["mae"]
    delta_acc = metrics_xgb["acc_wdl"] - metrics_poi["acc_wdl"]

    print(f"\n{sep}")
    print(f"  COMPARAÇÃO FINAL — TESTE (pós {CUTOFF})")
    print(sep)
    print(f"  {'Métrica':30s} {'XGBoost':>10}  {'Poisson':>10}  {'Delta':>8}")
    print("  " + "-" * 58)
    print(f"  {'MAE (lambda vs gols reais)':30s} {fmt(metrics_xgb['mae']):>10}  {fmt(metrics_poi['mae']):>10}  {delta_mae:>+8.4f}")
    print(f"  {'RMSE':30s} {fmt(metrics_xgb['rmse']):>10}  {fmt(metrics_poi['rmse']):>10}  {metrics_xgb['rmse']-metrics_poi['rmse']:>+8.4f}")
    print(f"  {'Acurácia W/D/L':30s} {fmt(metrics_xgb['acc_wdl'], True):>10}  {fmt(metrics_poi['acc_wdl'], True):>10}  {delta_acc:>+8.1%}")
    print(f"  {'Acurácia Vitória Casa (H)':30s} {fmt(metrics_xgb.get('acc_H'), True):>10}  {fmt(metrics_poi.get('acc_H'), True):>10}")
    print(f"  {'Acurácia Empate (D)':30s} {fmt(metrics_xgb.get('acc_D'), True):>10}  {fmt(metrics_poi.get('acc_D'), True):>10}")
    print(f"  {'Acurácia Vitória Visit. (A)':30s} {fmt(metrics_xgb.get('acc_A'), True):>10}  {fmt(metrics_poi.get('acc_A'), True):>10}")
    print(f"\n  Distribuição real (H/D/A): "
          f"H={metrics_xgb.get('n_H',0)}  D={metrics_xgb.get('n_D',0)}  A={metrics_xgb.get('n_A',0)}")

    print(f"\n{sep}")
    print("  TOP-10 FEATURES — SHAP (XGBoost v8)")
    print(sep)
    for rank, (feat, val) in enumerate(shap_top.head(10).items(), 1):
        bar    = "█" * int(val / shap_top.iloc[0] * 20)
        marker = " ← novo" if feat not in POISSON_FEATURES else ""
        print(f"  {rank:2d}. {feat:32s} {val:.4f}  {bar}{marker}")

    print(f"\n{sep}")
    print("  COMPARAÇÃO DE VERSÕES — XGBoost")
    print(sep)
    print(f"  {'Versão':28s} {'MAE':>8}  {'Acc WDL':>9}  {'ΔMAE':>8}  {'ΔAcc':>8}")
    print("  " + "-" * 65)
    prev_mae = prev_acc = None
    for ver, m in VERSION_HISTORY.items():
        d_mae = f"{m['mae_xgb'] - prev_mae:+.4f}" if prev_mae else "  base"
        d_acc = f"{m['acc_xgb'] - prev_acc:+.1%}" if prev_acc else "  base"
        print(f"  {ver:28s} {m['mae_xgb']:8.4f}  {m['acc_xgb']:9.1%}  {d_mae:>8}  {d_acc:>8}")
        prev_mae, prev_acc = m["mae_xgb"], m["acc_xgb"]

    v8_label = f"v8 (redesign ξ={best_xi:.3f})"
    d_mae_v8 = f"{metrics_xgb['mae'] - prev_mae:+.4f}" if prev_mae else "  base"
    d_acc_v8 = f"{metrics_xgb['acc_wdl'] - prev_acc:+.1%}" if prev_acc else "  base"
    print(f"  {v8_label:28s} {metrics_xgb['mae']:8.4f}  {metrics_xgb['acc_wdl']:9.1%}  {d_mae_v8:>8}  {d_acc_v8:>8}")
    print(sep)

    verdict = ""
    if delta_mae < -0.05 and delta_acc > 0.02:
        verdict = "XGBoost superior em MAE e acurácia — troca recomendada."
    elif delta_mae < -0.02:
        verdict = "XGBoost com MAE menor — ganho marginal."
    elif delta_acc > 0.03:
        verdict = "XGBoost com acurácia W/D/L melhor."
    elif delta_mae > 0.02 and delta_acc < 0.0:
        verdict = "Poisson mantém vantagem — XGBoost não agrega com features atuais."
    else:
        verdict = "Modelos equivalentes — melhoria marginal."
    print(f"\n  VEREDICTO: {verdict}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 62
    print(f"\n{sep}")
    print("  model_xgboost.py — Copa 2026 (v10: opp_saves_decay restaurado, 16 features)")
    print(f"  XGBoost count:poisson  vs  GLM Poisson")
    print(f"  Cutoff: {CUTOFF}")
    print(f"{sep}\n")

    # ── 1. Otimização de ξ ────────────────────────────────────────────────
    best_xi = tune_xi(XI_VALUES)

    # ── 2. Construir dataset final com ξ ótimo e salvar ──────────────────
    try:
        import build_features as bf
        print(f"── Construindo dataset final com ξ={best_xi:.3f} ──")
        df_raw  = bf.load_and_merge_raw()
        df_dec  = bf.add_decay_features(df_raw, best_xi)
        bf._print_decay_validation(df_dec, best_xi)
        df_imp  = bf.impute_by_confederation(df_dec)
        wide    = bf.build_match_dataset(df_imp)
        bf.save_data(wide)
    except ImportError:
        print("  build_features indisponível — lendo INPUT_CSV existente")
        wide = pd.read_csv(INPUT_CSV, parse_dates=["date"])

    # ── 3. Filtrar linhas com NaN em features core ───────────────────────
    core_cols = [c for c in wide.columns
                 if c.endswith("_decay") and not c.startswith("has_")]
    wide = wide.dropna(subset=core_cols).reset_index(drop=True)
    long = wide_to_long(wide)

    print(f"\n  Dataset: {len(wide)} jogos × {len(wide.columns)} cols")
    print(f"  Long format: {len(long)} linhas  |  Features XGB: {len(FEATURES)}")
    print(f"  Features: {', '.join(FEATURES)}\n")

    # ── 4. CV principal ───────────────────────────────────────────────────
    print("── Expanding-window CV ──")
    fold_results, cv_preds_xgb, cv_preds_poi = run_cv(wide, long)

    cv_metrics_xgb = evaluate_games(cv_preds_xgb)
    cv_metrics_poi = evaluate_games(cv_preds_poi)
    print(f"\n  CV Global — MAE XGB={cv_metrics_xgb['mae']:.4f} | MAE Poi={cv_metrics_poi['mae']:.4f}")

    # ── 5. Teste final (pós cutoff) ───────────────────────────────────────
    print(f"\n── Teste final (pós cutoff) ──")
    xgb_model, train_long, test_preds_xgb, test_preds_poi = run_final_test(wide, long)

    if test_preds_xgb:
        test_metrics_xgb = evaluate_games(test_preds_xgb)
        test_metrics_poi = evaluate_games(test_preds_poi)
        print_score_dist_table(test_preds_xgb, test_preds_poi)
    else:
        print("\n  Sem jogos de teste no período pós-cutoff.")
        print("  Usando métricas do CV global como referência.")
        test_metrics_xgb = cv_metrics_xgb
        test_metrics_poi = cv_metrics_poi

    # ── 6. SHAP ───────────────────────────────────────────────────────────
    print(f"\n── SHAP values ──")
    shap_top = compute_and_plot_shap(xgb_model, train_long)

    # ── 7. Salvar modelo v5 ───────────────────────────────────────────────
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"  Modelo salvo: {OUT_PKL}")

    # ── 8. Salvar CSV e tabela final ──────────────────────────────────────
    save_comparison_csv(fold_results, test_metrics_xgb, test_metrics_poi)
    print_final_table(test_metrics_xgb, test_metrics_poi, shap_top, best_xi)


if __name__ == "__main__":
    main()
