"""
model_poisson.py — Regressao de Poisson para predicao de gols (Copa 2026).

Features (modelo atual):
  gls_avg5, sot_avg5, tklw_avg5_adv, int_avg5_adv, win_rate_avg5, home_advantage

  win_rate_avg5 substituiu rank_diff (ajuste 1):
    - win_rate: % de vitórias nos ultimos 5 jogos (vitoria=1, empate=0.5, derrota=0)
    - rank_diff: removido das features de treino (mantido no dataset para referencia)

  host_factor (ajuste 2): documentado no dataset, nao usado no treino.
    Na Copa 2026: MEX=1.0/0.5, USA=1.0/0.5, CAN=1.0/0.5 dependendo do pais sede.

Validacao: expanding window cronologica (treino minimo 30 jogos, folds de 5 jogos).
Saidas:
  outputs/poisson_summary.txt       — summary GLM do modelo final
  outputs/poisson_predictions.csv   — predicoes no conjunto de teste
  outputs/poisson_cv_results.csv    — metricas por fold (modelo atual)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
INPUT_CSV   = Path("data/processed/model_dataset.csv")
OUTPUT_DIR  = Path("outputs")
SUMMARY_TXT = OUTPUT_DIR / "poisson_summary.txt"
PRED_CSV    = OUTPUT_DIR / "poisson_predictions.csv"
CV_CSV      = OUTPUT_DIR / "poisson_cv_results.csv"

CUTOFF     = "2025-09-30"   # split final treino/teste
MIN_TRAIN  = 30             # minimo de jogos (linhas long = 2x) para 1o fold
FOLD_SIZE  = 5              # jogos adicionados por fold

FEATURES = [
    "gls_avg5",            # media de gols marcados (forca ofensiva)
    "shots_on_goal_avg5",  # chutes ao gol — eficiencia ofensiva
    "win_rate_avg5",       # aproveitamento recente (forma)
    "opp_saves_avg5",      # saves medios do adversario (forca defensiva adversaria)
    "home_advantage",      # mando de campo
]

# Resultados do modelo anterior (win_rate_avg5 com tklw/int) — linha de base
PREV_MODEL_CV = {
    "label":        "v_anterior",
    "mae_mean":     0.8834,
    "rmse_mean":    1.1085,
    "acc_mean":     0.4667,
    "exact_total":  10,
    "total_games":  58,
}


# ---------------------------------------------------------------------------
# Carregamento e filtragem
# ---------------------------------------------------------------------------
def load_complete(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # Filtra jogos onde TODOS os avg5 e win_rate_avg5 estao preenchidos
    need_cols = [c for c in df.columns if c.endswith("_avg5")]
    df = df.dropna(subset=need_cols).reset_index(drop=True)
    print(f"Jogos com features completas (avg5 + win_rate): {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Wide -> Long  (2 linhas por jogo, perspectiva de cada time)
# ---------------------------------------------------------------------------
def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for side in ("home", "away"):
            opp = "away" if side == "home" else "home"
            rows.append({
                "match_id":           r["match_id"],
                "date":               r["date"],
                "home_team":          r["home_team"],
                "away_team":          r["away_team"],
                "team":               r[f"{side}_team"],
                "opponent":           r[f"{opp}_team"],
                "gols_marcados":      r[f"{side}_gols"],
                "gls_avg5":           r[f"{side}_gls_avg5"],
                "shots_on_goal_avg5": r[f"{side}_shots_on_goal_avg5"],
                "win_rate_avg5":      r[f"{side}_win_rate_avg5"],
                "opp_saves_avg5":     r[f"{opp}_saves_avg5"],
                "home_advantage":     1 if side == "home" else 0,
            })
    long = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"Long format: {len(long)} linhas ({len(df)} jogos x 2 times)")
    return long


# ---------------------------------------------------------------------------
# Treino e predicao do GLM Poisson
# ---------------------------------------------------------------------------
def fit_poisson(train: pd.DataFrame):
    X = sm.add_constant(train[FEATURES])
    y = train["gols_marcados"].astype(float)
    model = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.Log()))
    return model.fit(disp=False)


def predict_lambda(result, df: pd.DataFrame) -> pd.Series:
    X = sm.add_constant(df[FEATURES], has_constant="add")
    return result.predict(X)


def score_matrix(lh: float, la: float, max_goals: int = 8) -> tuple[int, int]:
    best_p, best_h, best_a = -1.0, 0, 0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            if p > best_p:
                best_p, best_h, best_a = p, h, a
    return best_h, best_a


def result_label(h: int | float, a: int | float) -> str:
    return "H" if h > a else ("A" if h < a else "D")


def evaluate_fold(result, test_long: pd.DataFrame, test_wide: pd.DataFrame) -> list[dict]:
    test_long = test_long.copy()
    test_long["lambda"] = predict_lambda(result, test_long)

    preds = []
    for _, g in test_wide.iterrows():
        mid = g["match_id"]
        home_row = test_long[(test_long["match_id"] == mid) &
                             (test_long["team"] == g["home_team"])]
        away_row = test_long[(test_long["match_id"] == mid) &
                             (test_long["team"] == g["away_team"])]
        if home_row.empty or away_row.empty:
            continue
        lh = float(home_row["lambda"].iloc[0])
        la = float(away_row["lambda"].iloc[0])
        ph, pa = score_matrix(lh, la)
        rh, ra = int(g["home_gols"]), int(g["away_gols"])
        preds.append({
            "match_id":           mid,
            "date":               g["date"].date(),
            "home_team":          g["home_team"],
            "away_team":          g["away_team"],
            "home_gols_real":     rh,
            "away_gols_real":     ra,
            "home_lambda":        round(lh, 3),
            "away_lambda":        round(la, 3),
            "home_gols_previsto": ph,
            "away_gols_previsto": pa,
            "placar_previsto":    f"{ph}-{pa}",
            "placar_real":        f"{rh}-{ra}",
            "resultado_correto":  result_label(ph, pa) == result_label(rh, ra),
        })
    return preds


# ---------------------------------------------------------------------------
# Cross-validation temporal expanding window
# ---------------------------------------------------------------------------
def expanding_window_cv(long: pd.DataFrame, wide: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Itera por folds cronologicos com janela expansiva.
    Retorna: (lista de predicoes por jogo, lista de metricas por fold)
    """
    games_sorted = wide.sort_values("date").reset_index(drop=True)
    n_games      = len(games_sorted)

    all_preds   = []
    fold_metrics = []
    fold_num    = 0

    train_end = MIN_TRAIN - 1  # indice do ultimo jogo de treino

    while train_end + 1 < n_games:
        test_end = min(train_end + FOLD_SIZE, n_games - 1)

        train_games = games_sorted.iloc[: train_end + 1]
        test_games  = games_sorted.iloc[train_end + 1 : test_end + 1]

        train_ids   = set(train_games["match_id"])
        test_ids    = set(test_games["match_id"])

        train_long  = long[long["match_id"].isin(train_ids)]
        test_long   = long[long["match_id"].isin(test_ids)]

        if len(train_long) < 10 or len(test_long) == 0:
            train_end = test_end
            continue

        try:
            result = fit_poisson(train_long)
        except Exception as e:
            print(f"  Fold {fold_num + 1}: erro no ajuste — {e}")
            train_end = test_end
            continue

        # Predicoes no long format (necessario para MAE/RMSE por jogo)
        test_long_pred = test_long.copy()
        test_long_pred["lambda"] = predict_lambda(result, test_long_pred)

        preds = evaluate_fold(result, test_long_pred, test_games)
        if not preds:
            train_end = test_end
            continue

        fold_num += 1
        for p in preds:
            p["fold"] = fold_num
        all_preds.extend(preds)

        # Metricas do fold
        pred_df = pd.DataFrame(preds)
        mae     = float(np.mean(np.abs(test_long_pred["lambda"] - test_long_pred["gols_marcados"])))
        rmse    = float(np.sqrt(np.mean((test_long_pred["lambda"] - test_long_pred["gols_marcados"]) ** 2)))
        acc     = pred_df["resultado_correto"].mean()
        exact   = int((pred_df["placar_previsto"] == pred_df["placar_real"]).sum())

        fold_metrics.append({
            "fold":         fold_num,
            "train_games":  len(train_games),
            "test_games":   len(test_games),
            "mae":          round(mae, 4),
            "rmse":         round(rmse, 4),
            "acc_wdl":      round(acc, 4),
            "exact_scores": exact,
            "train_cutoff": str(train_games["date"].max().date()),
            "test_period":  f"{test_games['date'].min().date()} / {test_games['date'].max().date()}",
        })

        train_end = test_end

    return all_preds, fold_metrics


# ---------------------------------------------------------------------------
# Interpretacao dos coeficientes
# ---------------------------------------------------------------------------
def interpret_coefficients(result) -> None:
    desc_map = {
        "gls_avg5":            "gols medios marcados (ataque proprio)",
        "shots_on_goal_avg5":  "chutes ao gol medios (eficiencia ofensiva)",
        "win_rate_avg5":       "% vitorias ultimos 5 jogos (forma)",
        "opp_saves_avg5":      "saves medios do adversario (defesa adversaria)",
        "home_advantage":      "vantagem de mando (1=casa, 0=fora)",
    }
    params = result.params
    conf   = result.conf_int()
    print(f"\n  {'Feature':<22s} {'exp(b)':>8s} {'IC95%Lo':>8s} {'IC95%Hi':>8s} {'p':>7s}")
    print("  " + "-" * 62)
    ns_features = []
    for feat in FEATURES:
        if feat not in params.index:
            continue
        exp_c = np.exp(params[feat])
        ci_lo = np.exp(conf.loc[feat, 0])
        ci_hi = np.exp(conf.loc[feat, 1])
        pval  = result.pvalues[feat]
        sig   = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "ns"))
        print(f"  {feat:<22s} {exp_c:8.3f} {ci_lo:8.3f} {ci_hi:8.3f} {pval:7.3f}{sig}")
        print(f"    '{desc_map.get(feat, feat)}': mult. gols por {exp_c:.3f}")
        if pval > 0.10:
            ns_features.append((feat, pval))

    if ns_features:
        print(f"\n  ATENCAO — features nao significativas (p > 0.10):")
        for feat, pval in ns_features:
            print(f"    {feat}: p={pval:.3f} — considerar remocao do modelo")


def print_comparison_table(prev: dict, curr_cv: pd.DataFrame) -> None:
    """Imprime tabela lado a lado: modelo anterior (rank_diff) vs atual (win_rate_avg5)."""
    curr_exact  = int(curr_cv["exact_scores"].sum())
    curr_games  = int(curr_cv["test_games"].sum())
    curr_mae    = curr_cv["mae"].mean()
    curr_rmse   = curr_cv["rmse"].mean()
    curr_acc    = curr_cv["acc_wdl"].mean()

    def delta(curr, prev, invert=False):
        d = curr - prev
        better = d < 0 if not invert else d > 0
        arrow = "v" if d < 0 else "^"
        mark  = " (melhor)" if better else " (pior)"
        return f"{d:+.4f} {arrow}{mark}"

    print(f"\n  {'Metrica':<26s} {'rank_diff':>12s} {'win_rate_avg5':>14s} {'Delta':>30s}")
    print("  " + "-" * 85)
    print(f"  {'MAE (media folds)':<26s} {prev['mae_mean']:12.4f} {curr_mae:14.4f} {delta(curr_mae, prev['mae_mean'])}")
    print(f"  {'RMSE (media folds)':<26s} {prev['rmse_mean']:12.4f} {curr_rmse:14.4f} {delta(curr_rmse, prev['rmse_mean'])}")
    print(f"  {'Acc W/D/L (media)':<26s} {prev['acc_mean']:12.1%} {curr_acc:14.1%} {delta(curr_acc, prev['acc_mean'], invert=True)}")
    exact_prev = f"{prev['exact_total']}/{prev['total_games']}"
    exact_curr = f"{curr_exact}/{curr_games}"
    print(f"  {'Placares exatos':<26s} {exact_prev:>12s} {exact_curr:>14s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  model_poisson.py — GLM Poisson + Expanding Window CV")
    print("=" * 70 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carrega
    wide = load_complete(INPUT_CSV)
    long = wide_to_long(wide)

    # 2. Cross-validation temporal
    print("\n" + "=" * 70)
    print("  CROSS-VALIDATION TEMPORAL (expanding window)")
    print(f"  Treino minimo: {MIN_TRAIN} jogos | Fold size: {FOLD_SIZE} jogos")
    print("=" * 70)

    all_preds, fold_metrics = expanding_window_cv(long, wide)

    cv_df = pd.DataFrame(fold_metrics)
    print(f"\n  {len(cv_df)} folds executados\n")

    if not cv_df.empty:
        print(cv_df[["fold", "train_games", "test_games", "mae", "rmse",
                     "acc_wdl", "exact_scores", "test_period"]].to_string(index=False))
        print(f"\n  Media dos folds:")
        print(f"    MAE        : {cv_df['mae'].mean():.4f}  (std={cv_df['mae'].std():.4f})")
        print(f"    RMSE       : {cv_df['rmse'].mean():.4f}  (std={cv_df['rmse'].std():.4f})")
        print(f"    Acc W/D/L  : {cv_df['acc_wdl'].mean():.1%}  (std={cv_df['acc_wdl'].std():.1%})")
        total_exact = cv_df["exact_scores"].sum()
        total_games = cv_df["test_games"].sum()
        print(f"    Placares exatos: {total_exact}/{total_games}")
        cv_df.to_csv(CV_CSV, index=False)
        print(f"\n  CV results salvo em: {CV_CSV}")

    # 3. Modelo final (treino em todos os jogos até CUTOFF)
    train_games = wide[wide["date"] <= CUTOFF]
    test_games  = wide[wide["date"] >  CUTOFF]
    train_long  = long[long["date"] <= CUTOFF]
    test_long   = long[long["date"] >  CUTOFF]

    print(f"\n{'=' * 70}")
    print(f"  MODELO FINAL — treino ate {CUTOFF}")
    print(f"  Treino: {len(train_games)} jogos | Teste: {len(test_games)} jogos")
    print("=" * 70)

    result_final = fit_poisson(train_long)
    print(result_final.summary())

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(str(result_final.summary()))
    print(f"\n  Summary salvo em: {SUMMARY_TXT}")

    # 4. Interpretacao dos coeficientes
    print("\n" + "=" * 70)
    print("  INTERPRETACAO DOS COEFICIENTES")
    print("=" * 70)
    interpret_coefficients(result_final)

    # 5. Predicoes no conjunto de teste (modelo final)
    if len(test_games) > 0:
        test_long_pred = test_long.copy()
        test_long_pred["lambda"] = predict_lambda(result_final, test_long_pred)
        preds_final = evaluate_fold(result_final, test_long_pred, test_games)
        if preds_final:
            pred_df = pd.DataFrame(preds_final)
            print(f"\n{'=' * 70}")
            print("  PREDICOES NO CONJUNTO DE TESTE (modelo final)")
            print("=" * 70)
            cols = ["date", "home_team", "away_team",
                    "home_lambda", "away_lambda",
                    "placar_previsto", "placar_real", "resultado_correto"]
            print(pred_df[cols].to_string(index=False))

            mae  = float(np.mean(np.abs(test_long_pred["lambda"] - test_long_pred["gols_marcados"])))
            rmse = float(np.sqrt(np.mean((test_long_pred["lambda"] - test_long_pred["gols_marcados"]) ** 2)))
            acc  = pred_df["resultado_correto"].mean()
            exact = int((pred_df["placar_previsto"] == pred_df["placar_real"]).sum())

            pred_df.to_csv(PRED_CSV, index=False)
            print(f"\n  Predicoes salvas em: {PRED_CSV}")
    else:
        mae = rmse = acc = exact = float("nan")
        preds_final = []

    # 6. Feature mais importante
    std_feats = train_long[FEATURES].std()
    scaled    = (result_final.params[FEATURES].abs() * std_feats).sort_values(ascending=False)
    top_feat  = scaled.idxmax()

    # 7. Comparacao com modelo anterior
    if not cv_df.empty:
        print(f"\n{'=' * 70}")
        print("  COMPARACAO: rank_diff vs win_rate_avg5")
        print("=" * 70)
        print_comparison_table(PREV_MODEL_CV, cv_df)

    # 8. Resumo executivo
    print(f"\n{'=' * 70}")
    print("  RESUMO EXECUTIVO")
    print("=" * 70)
    if not cv_df.empty:
        print(f"  CV MAE  (media folds)  : {cv_df['mae'].mean():.4f}")
        print(f"  CV RMSE (media folds)  : {cv_df['rmse'].mean():.4f}")
        print(f"  CV Acc W/D/L (media)   : {cv_df['acc_wdl'].mean():.1%}")
        print(f"  CV Placares exatos     : {cv_df['exact_scores'].sum()}/{cv_df['test_games'].sum()}")
    print(f"  Feature mais importante: {top_feat}  (exp(b)={np.exp(result_final.params[top_feat]):.3f})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
