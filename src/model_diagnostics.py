"""
model_diagnostics.py — Diagnóstico do modelo Poisson para identificar se o problema
principal é nos dados (imputação) ou na modelagem (features/distribuição).

DIAGNÓSTICO 1 — MAE por confederação
DIAGNÓSTICO 2 — Impacto da imputação nos erros
DIAGNÓSTICO 3 — Multicolinearidade entre features
DIAGNÓSTICO 4 — Superdispersão de gols por confederação

Saída: outputs/model_diagnostics.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import poisson

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INPUT_CSV  = Path("data/processed/model_dataset.csv")
OUTPUT_TXT = Path("outputs/model_diagnostics.txt")
OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

CUTOFF = "2025-09-30"

FEATURES = [
    "gls_avg5",
    "shots_on_goal_avg5",
    "win_rate_avg5",
    "opp_saves_avg5",
    "home_advantage",
]

ALL_AVG5_LONG = [
    "gls_avg5", "shots_avg5", "shots_on_goal_avg5", "blocked_shots_avg5",
    "ball_possession_avg5", "fouls_avg5", "corners_avg5",
    "passes_accurate_avg5", "saves_avg5", "win_rate_avg5",
]

# Mapeamento time → confederação (48 qualificadas)
TEAM_CONF = {}
QUALIFIED_48 = {
    "UEFA":     ["Austria","Belgium","Bosnia and Herzegovina","Croatia","Czech Republic",
                 "England","France","Germany","Netherlands","Norway","Portugal",
                 "Scotland","Spain","Sweden","Switzerland","Turkey"],
    "CAF":      ["Algeria","Cape Verde","DR Congo","Egypt","Ghana","Ivory Coast",
                 "Morocco","Senegal","South Africa","Tunisia"],
    "AFC":      ["Australia","Iran","Iraq","Japan","Jordan","Qatar",
                 "Saudi Arabia","South Korea","Uzbekistan"],
    "CONCACAF": ["Canada","Curacao","Haiti","Mexico","Panama","United States"],
    "CONMEBOL": ["Argentina","Brazil","Colombia","Ecuador","Paraguay","Uruguay"],
    "OFC":      ["New Zealand"],
}
for conf, teams in QUALIFIED_48.items():
    for t in teams:
        TEAM_CONF[t] = conf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    need = [c for c in df.columns if c.endswith("_avg5")]
    df = df.dropna(subset=need).reset_index(drop=True)
    return df


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for side, opp in [("home", "away"), ("away", "home")]:
            rows.append({
                "match_id":           r["match_id"],
                "date":               r["date"],
                "match_type":         r.get("match_type", ""),
                "team":               r[f"{side}_team"],
                "opponent":           r[f"{opp}_team"],
                "gols_marcados":      r[f"{side}_gols"],
                "imputed":            bool(r.get(f"{side}_imputed", False)),
                "opp_imputed":        bool(r.get(f"{opp}_imputed", False)),
                # features
                "gls_avg5":           r[f"{side}_gls_avg5"],
                "shots_avg5":         r[f"{side}_shots_avg5"],
                "shots_on_goal_avg5": r[f"{side}_shots_on_goal_avg5"],
                "blocked_shots_avg5": r[f"{side}_blocked_shots_avg5"],
                "ball_possession_avg5": r[f"{side}_ball_possession_avg5"],
                "fouls_avg5":         r[f"{side}_fouls_avg5"],
                "corners_avg5":       r[f"{side}_corners_avg5"],
                "passes_accurate_avg5": r[f"{side}_passes_accurate_avg5"],
                "saves_avg5":         r[f"{side}_saves_avg5"],
                "win_rate_avg5":      r[f"{side}_win_rate_avg5"],
                "opp_saves_avg5":     r[f"{opp}_saves_avg5"],
                "home_advantage":     1 if side == "home" else 0,
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fit_poisson(train_long: pd.DataFrame):
    X = sm.add_constant(train_long[FEATURES])
    y = train_long["gols_marcados"].astype(float)
    return sm.GLM(y, X, family=sm.families.Poisson(
        link=sm.families.links.Log())).fit(disp=False)


def predict_lambdas(result, df: pd.DataFrame) -> pd.Series:
    X = sm.add_constant(df[FEATURES], has_constant="add")
    return result.predict(X)


def score_matrix(lh: float, la: float, max_g: int = 8) -> tuple[int, int]:
    best_p, bh, ba = -1.0, 0, 0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            if p > best_p:
                best_p, bh, ba = p, h, a
    return bh, ba


def result_label(h, a):
    return "H" if h > a else ("A" if h < a else "D")


def mae(actual, predicted):
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def acc_wdl(actual_h, actual_a, pred_h, pred_a):
    correct = sum(
        result_label(ah, aa) == result_label(ph, pa)
        for ah, aa, ph, pa in zip(actual_h, actual_a, pred_h, pred_a)
    )
    return correct / len(actual_h) if actual_h else float("nan")


# ---------------------------------------------------------------------------
# Gera predições no conjunto de teste (modelo treinado no train)
# ---------------------------------------------------------------------------

def get_test_predictions(wide: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    train_w = wide[wide["date"] <= CUTOFF]
    test_w  = wide[wide["date"] >  CUTOFF]
    train_l = long[long["date"] <= CUTOFF]
    test_l  = long[long["date"] >  CUTOFF].copy()

    result = fit_poisson(train_l)
    test_l["lambda"] = predict_lambdas(result, test_l)

    # Merge home/away lambda into test_w
    preds = []
    for _, g in test_w.iterrows():
        mid = g["match_id"]
        hr = test_l[(test_l["match_id"] == mid) & (test_l["team"] == g["home_team"])]
        ar = test_l[(test_l["match_id"] == mid) & (test_l["team"] == g["away_team"])]
        if hr.empty or ar.empty:
            continue
        lh, la = float(hr["lambda"].iloc[0]), float(ar["lambda"].iloc[0])
        ph, pa = score_matrix(lh, la)
        preds.append({
            "match_id":         mid,
            "date":             g["date"],
            "home_team":        g["home_team"],
            "away_team":        g["away_team"],
            "home_gols":        int(g["home_gols"]),
            "away_gols":        int(g["away_gols"]),
            "home_lambda":      round(lh, 4),
            "away_lambda":      round(la, 4),
            "pred_home":        ph,
            "pred_away":        pa,
            "home_imputed":     bool(g.get("home_imputed", False)),
            "away_imputed":     bool(g.get("away_imputed", False)),
            # confederação de cada time
            "home_conf":        TEAM_CONF.get(g["home_team"], "OTHER"),
            "away_conf":        TEAM_CONF.get(g["away_team"], "OTHER"),
            # erro individual por time (long-format error)
            "home_mae":         abs(lh - int(g["home_gols"])),
            "away_mae":         abs(la - int(g["away_gols"])),
            "result_correct":   result_label(ph, pa) == result_label(int(g["home_gols"]), int(g["away_gols"])),
        })

    pred_df = pd.DataFrame(preds)

    # Adiciona flag de grupo de imputação (nível de jogo)
    pred_df["any_imputed"]  = pred_df["home_imputed"] | pred_df["away_imputed"]
    pred_df["both_real"]    = ~pred_df["any_imputed"]
    pred_df["same_conf"]    = pred_df["home_conf"] == pred_df["away_conf"]

    return pred_df, result


# ---------------------------------------------------------------------------
# DIAGNÓSTICO 1 — MAE por confederação
# ---------------------------------------------------------------------------

def diag1_mae_by_conf(pred_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  DIAGNÓSTICO 1 — MAE / Acurácia W/D/L por Confederação")
    lines.append("=" * 70)

    # Prepara long do test para MAE por time
    # (pred_df tem 1 linha por jogo — vamos decompor home/away)
    rows_by_conf = {}
    for _, r in pred_df.iterrows():
        for side, conf, lam, real, imp in [
            ("home", r["home_conf"], r["home_lambda"], r["home_gols"], r["home_imputed"]),
            ("away", r["away_conf"], r["away_lambda"], r["away_gols"], r["away_imputed"]),
        ]:
            if conf not in rows_by_conf:
                rows_by_conf[conf] = {"lambdas": [], "reals": [], "imputed_flags": []}
            rows_by_conf[conf]["lambdas"].append(lam)
            rows_by_conf[conf]["reals"].append(real)
            rows_by_conf[conf]["imputed_flags"].append(imp)

    lines.append(f"\n  {'Conf':10s} {'N_jogos':>8} {'MAE':>7} {'RMSE':>7} "
                 f"{'Acc W/D/L':>10} {'% imputado':>11} {'Sinal'}")
    lines.append("  " + "-" * 66)

    overall_mae_vals = []
    conf_mae = {}

    # Acc por jogo (wide level)
    acc_by_conf = {}
    for _, r in pred_df.iterrows():
        for conf in [r["home_conf"], r["away_conf"]]:
            if conf not in acc_by_conf:
                acc_by_conf[conf] = {"correct": 0, "total": 0}
            acc_by_conf[conf]["total"] += 1
            if r["result_correct"]:
                acc_by_conf[conf]["correct"] += 1

    for conf in sorted(rows_by_conf.keys()):
        d = rows_by_conf[conf]
        lams = np.array(d["lambdas"])
        reals = np.array(d["reals"])
        imp_flags = np.array(d["imputed_flags"])
        n = len(lams)
        m = mae(reals, lams)
        r = rmse(reals, lams)
        pct_imp = imp_flags.mean() * 100
        acc_d = acc_by_conf.get(conf, {})
        acc_val = acc_d.get("correct", 0) / acc_d.get("total", 1)
        n_games = acc_d.get("total", 0)

        conf_mae[conf] = m
        overall_mae_vals.append(m)

        flag = ""
        if pct_imp > 50:
            flag = "<-- ALTO %imputado"
        elif m > 1.3:
            flag = "<-- MAE elevado"

        lines.append(f"  {conf:10s} {n_games:8d} {m:7.4f} {r:7.4f} "
                     f"{acc_val:10.1%} {pct_imp:10.1f}%  {flag}")

    # Comparação direta: CONMEBOL+UEFA vs CAF+OFC
    good_confs = ["CONMEBOL", "UEFA"]
    weak_confs = ["CAF", "OFC"]
    good_mae_vals = [v for c, v in conf_mae.items() if c in good_confs]
    weak_mae_vals = [v for c, v in conf_mae.items() if c in weak_confs]
    if good_mae_vals and weak_mae_vals:
        delta = np.mean(weak_mae_vals) - np.mean(good_mae_vals)
        lines.append(f"\n  Delta MAE (CAF+OFC vs CONMEBOL+UEFA): {delta:+.4f}")
        lines.append(f"  {'Maior MAE em CAF/OFC confirma impacto dos dados' if delta > 0.1 else 'Diferença pequena — problema não é só nos dados'}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DIAGNÓSTICO 2 — Impacto da imputação
# ---------------------------------------------------------------------------

def diag2_imputation_impact(pred_df: pd.DataFrame) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  DIAGNÓSTICO 2 — Impacto da Imputação nos Erros")
    lines.append("=" * 70)

    def group_metrics(mask, label):
        sub = pred_df[mask]
        if sub.empty:
            return f"  {label}: sem dados"
        errors = list(sub["home_mae"]) + list(sub["away_mae"])
        lambdas = list(sub["home_lambda"]) + list(sub["away_lambda"])
        reals   = list(sub["home_gols"])   + list(sub["away_gols"])
        m  = mae(reals, lambdas)
        r  = rmse(reals, lambdas)
        ac = sub["result_correct"].mean()
        n  = len(sub)
        return (f"  {label:40s}: N={n:4d}  MAE={m:.4f}  RMSE={r:.4f}  "
                f"Acc={ac:.1%}")

    lines.append("")
    lines.append(group_metrics(pred_df["both_real"],   "Ambos os times SEM imputação"))
    lines.append(group_metrics(pred_df["any_imputed"], "Pelo menos 1 time COM imputação"))
    lines.append("")
    lines.append(group_metrics(pred_df["home_imputed"], "Time da casa imputado"))
    lines.append(group_metrics(pred_df["away_imputed"], "Time visitante imputado"))

    # Quantitativo
    r_mask = pred_df["both_real"]
    i_mask = pred_df["any_imputed"]

    def _mae_group(mask):
        sub = pred_df[mask]
        if sub.empty:
            return float("nan")
        lams  = list(sub["home_lambda"]) + list(sub["away_lambda"])
        reals = list(sub["home_gols"])   + list(sub["away_gols"])
        return mae(reals, lams)

    m_real = _mae_group(r_mask)
    m_imp  = _mae_group(i_mask)

    if not np.isnan(m_real) and not np.isnan(m_imp):
        ratio = m_imp / m_real
        lines.append(f"\n  Razão MAE(imputado) / MAE(real): {ratio:.3f}")
        if ratio > 1.3:
            lines.append("  CONCLUSÃO: imputação está degradando significativamente as previsões (ratio > 1.3)")
            lines.append("  → Problema predominantemente de DADOS")
        elif ratio > 1.1:
            lines.append("  CONCLUSÃO: imputação tem impacto moderado (ratio 1.1-1.3)")
            lines.append("  → Problema misto dados + modelo")
        else:
            lines.append("  CONCLUSÃO: imputação não é a causa principal (ratio ≤ 1.1)")
            lines.append("  → Problema predominantemente de MODELAGEM")

    # Teste estatístico: Mann-Whitney U (erros real vs imputado)
    errors_real = (
        list(pred_df.loc[r_mask, "home_mae"]) +
        list(pred_df.loc[r_mask, "away_mae"])
    )
    errors_imp = (
        list(pred_df.loc[i_mask, "home_mae"]) +
        list(pred_df.loc[i_mask, "away_mae"])
    )
    if errors_real and errors_imp:
        stat, pval = stats.mannwhitneyu(errors_real, errors_imp, alternative="less")
        lines.append(f"\n  Teste Mann-Whitney (real < imputado): U={stat:.0f}, p={pval:.4f}")
        lines.append(f"  {'Diferença estatisticamente significativa (p<0.05)' if pval < 0.05 else 'Diferença não significativa (p≥0.05)'}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DIAGNÓSTICO 3 — Multicolinearidade
# ---------------------------------------------------------------------------

def diag3_correlation(long_train: pd.DataFrame) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  DIAGNÓSTICO 3 — Multicolinearidade entre Features")
    lines.append("=" * 70)

    corr_cols = ALL_AVG5_LONG
    avail = [c for c in corr_cols if c in long_train.columns]
    corr = long_train[avail].corr(method="pearson")

    lines.append(f"\n  Pares com |r| > 0.7 (risco de colinearidade):")
    lines.append(f"  {'Feature A':30s} {'Feature B':30s} {'r':>7}")
    lines.append("  " + "-" * 68)
    found = False
    for i, fa in enumerate(avail):
        for fb in avail[i+1:]:
            r = corr.loc[fa, fb]
            if abs(r) > 0.7:
                flag = " <-- ALTA" if abs(r) > 0.85 else ""
                lines.append(f"  {fa:30s} {fb:30s} {r:7.3f}{flag}")
                found = True
    if not found:
        lines.append("  Nenhum par com |r| > 0.7 — sem multicolinearidade severa")

    # Foco especial: win_rate_avg5 vs gls_avg5
    if "win_rate_avg5" in corr.index and "gls_avg5" in corr.index:
        r_wg = corr.loc["win_rate_avg5", "gls_avg5"]
        lines.append(f"\n  Correlação win_rate_avg5 × gls_avg5: r = {r_wg:.3f}")
        if abs(r_wg) > 0.8:
            lines.append("  ALERTA: r > 0.8 → win_rate é redundante, adiciona colinearidade")
            lines.append("  → Considerar remover win_rate_avg5 do modelo")
        elif abs(r_wg) > 0.6:
            lines.append("  AVISO: r moderado (0.6-0.8) — colinearidade aceitável mas monitorar")
        else:
            lines.append("  OK: correlação baixa — ambas as features trazem informação independente")

    # VIF para as features do modelo
    lines.append(f"\n  VIF (Variance Inflation Factor) das features do modelo:")
    lines.append(f"  {'Feature':25s} {'VIF':>7} {'Sinal'}")
    lines.append("  " + "-" * 50)
    feat_data = long_train[FEATURES].dropna()
    for feat in FEATURES:
        other = [f for f in FEATURES if f != feat]
        X_other = sm.add_constant(feat_data[other])
        y_feat  = feat_data[feat]
        try:
            r2 = sm.OLS(y_feat, X_other).fit().rsquared
            vif = 1 / (1 - r2) if r2 < 1.0 else float("inf")
            flag = " <-- PROBLEMA" if vif > 10 else (" <-- AVISO" if vif > 5 else "")
            lines.append(f"  {feat:25s} {vif:7.2f}{flag}")
        except Exception:
            lines.append(f"  {feat:25s}   N/A")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DIAGNÓSTICO 4 — Superdispersão de gols por confederação
# ---------------------------------------------------------------------------

def diag4_overdispersion(long_train: pd.DataFrame) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  DIAGNÓSTICO 4 — Superdispersão de Gols por Confederação")
    lines.append("=" * 70)
    lines.append(f"\n  Poisson assume Var/Média = 1. Razão > 1.5 indica superdispersão.")
    lines.append(f"\n  {'Conf':12s} {'N':>6} {'Média':>7} {'Var':>8} {'Var/Média':>10} {'% zeros':>9} {'Poisson?'}")
    lines.append("  " + "-" * 72)

    # Adiciona confederação do time
    long_train = long_train.copy()
    long_train["team_conf"] = long_train["team"].map(lambda t: TEAM_CONF.get(t, "OTHER"))

    global_g = long_train["gols_marcados"].dropna()
    overall_disp = global_g.var() / global_g.mean()

    conf_disp = {}
    for conf, grp in long_train.groupby("team_conf"):
        g = grp["gols_marcados"].dropna()
        if len(g) < 20:
            continue
        m = g.mean()
        v = g.var()
        disp = v / m
        pct_zero = (g == 0).mean() * 100
        conf_disp[conf] = disp
        flag = "OK" if disp <= 1.5 else ("MODERADO" if disp <= 2.5 else "SEVERO")
        lines.append(f"  {conf:12s} {len(g):6d} {m:7.3f} {v:8.3f} {disp:10.3f} "
                     f"{pct_zero:8.1f}%  {flag}")

    lines.append(f"\n  Global (todos os jogos): Var/Média = {overall_disp:.3f}")

    # Teste formal de superdispersão (Cameron-Trivedi para Poisson)
    lines.append(f"\n  Teste de superdispersão (Cameron-Trivedi):")
    g_all = long_train["gols_marcados"].dropna()
    lam_hat = g_all.mean()  # usando média como proxy do lambda
    aux_var = (g_all - lam_hat) ** 2 - g_all
    aux_reg = sm.OLS(aux_var / lam_hat, sm.add_constant(g_all / lam_hat - 1)).fit()
    alpha = aux_reg.params.iloc[1]
    pval_disp = aux_reg.pvalues.iloc[1]
    lines.append(f"  alpha = {alpha:.4f}  p = {pval_disp:.4f}")
    if pval_disp < 0.05 and alpha > 0:
        lines.append("  CONCLUSÃO: superdispersão significativa (p<0.05, alpha>0)")
        lines.append("  → Considerar Negative Binomial em vez de Poisson")
    else:
        lines.append("  Sem evidência significativa de superdispersão")

    # Zero-inflation test
    lines.append(f"\n  Zeros observados vs esperados (Poisson com lambda=média):")
    for conf in sorted(long_train["team_conf"].unique()):
        g = long_train[long_train["team_conf"] == conf]["gols_marcados"].dropna()
        if len(g) < 20:
            continue
        lam = g.mean()
        obs_zeros  = (g == 0).sum()
        exp_zeros  = len(g) * poisson.pmf(0, lam)
        ratio = obs_zeros / exp_zeros if exp_zeros > 0 else float("nan")
        flag = " <-- ZERO-INFLATED" if ratio > 1.5 else ""
        lines.append(f"  {conf:12s}: obs={obs_zeros:4d}  exp={exp_zeros:6.1f}  "
                     f"ratio={ratio:.2f}{flag}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CONCLUSÃO INTEGRADA
# ---------------------------------------------------------------------------

def conclusao(pred_df: pd.DataFrame, long_train: pd.DataFrame) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  CONCLUSÃO INTEGRADA")
    lines.append("=" * 70)

    # Evidências coletadas automaticamente
    evidencias_dados = 0
    evidencias_modelo = 0

    # D1: diferença CAF vs CONMEBOL/UEFA
    conf_errors = {}
    for _, r in pred_df.iterrows():
        for side, conf in [("home", r["home_conf"]), ("away", r["away_conf"])]:
            lam  = r[f"{side}_lambda"]
            real = r[f"{side}_gols"]
            if conf not in conf_errors:
                conf_errors[conf] = []
            conf_errors[conf].append(abs(lam - real))

    mae_conmebol = np.mean(conf_errors.get("CONMEBOL", [float("nan")]))
    mae_caf      = np.mean(conf_errors.get("CAF", [float("nan")]))
    mae_uefa     = np.mean(conf_errors.get("UEFA", [float("nan")]))
    mae_afc      = np.mean(conf_errors.get("AFC", [float("nan")]))

    d1_delta = (mae_caf - ((mae_conmebol + mae_uefa) / 2)) if not np.isnan(mae_caf) else 0
    if d1_delta > 0.15:
        evidencias_dados += 2
    elif d1_delta > 0.05:
        evidencias_dados += 1

    # D2: ratio imputado/real
    r_mask = pred_df["both_real"]
    i_mask = pred_df["any_imputed"]
    def _m(mask):
        sub = pred_df[mask]
        if sub.empty: return float("nan")
        l = list(sub["home_lambda"]) + list(sub["away_lambda"])
        r = list(sub["home_gols"])   + list(sub["away_gols"])
        return mae(r, l)
    ratio_d2 = _m(i_mask) / _m(r_mask) if not np.isnan(_m(r_mask)) and _m(r_mask) > 0 else 1.0
    if ratio_d2 > 1.3:
        evidencias_dados += 2
    elif ratio_d2 > 1.1:
        evidencias_dados += 1
    else:
        evidencias_modelo += 1

    # D3: colinearidade win_rate vs gls
    avail = [c for c in ALL_AVG5_LONG if c in long_train.columns]
    corr = long_train[avail].corr()
    r_wg = corr.loc["win_rate_avg5", "gls_avg5"] if ("win_rate_avg5" in corr and "gls_avg5" in corr) else 0
    if abs(r_wg) > 0.8:
        evidencias_modelo += 2
    elif abs(r_wg) > 0.6:
        evidencias_modelo += 1

    # D4: superdispersão global
    g_all = long_train["gols_marcados"].dropna()
    overall_disp = g_all.var() / g_all.mean()
    if overall_disp > 2.0:
        evidencias_modelo += 2
    elif overall_disp > 1.5:
        evidencias_modelo += 1

    lines.append(f"\n  Evidências de problema nos DADOS  : {'█' * evidencias_dados} ({evidencias_dados} pontos)")
    lines.append(f"  Evidências de problema no MODELO  : {'█' * evidencias_modelo} ({evidencias_modelo} pontos)")
    lines.append("")

    if evidencias_dados > evidencias_modelo + 1:
        conclusao_final = "DADOS"
        detalhe = (
            "O erro é maior em jogos com imputação e/ou times com dados escassos (CAF/OFC).\n"
            "  Prioridade: coletar mais dados reais para CAF (AFCON Qualificação, amistosos\n"
            "  com statistics disponíveis) e NZ antes de ajustar o modelo."
        )
    elif evidencias_modelo > evidencias_dados + 1:
        conclusao_final = "MODELO"
        detalhe = (
            "A imputação não é a causa principal — o erro persiste mesmo em dados reais.\n"
            "  Prioridade: testar Negative Binomial (superdispersão), remover features\n"
            "  colineares, ou adicionar features defensivas independentes."
        )
    else:
        conclusao_final = "MISTO"
        detalhe = (
            "Ambos contribuem de forma comparável.\n"
            "  Recomendação: atacar dados primeiro (mais impacto/esforço) e\n"
            "  depois refinar o modelo."
        )

    lines.append(f"  DIAGNÓSTICO FINAL: problema predominantemente de {conclusao_final}")
    lines.append(f"\n  Detalhe: {detalhe}")

    lines.append(f"\n  Métricas de referência:")
    lines.append(f"    MAE CONMEBOL   : {mae_conmebol:.4f}")
    lines.append(f"    MAE UEFA       : {mae_uefa:.4f}")
    lines.append(f"    MAE AFC        : {mae_afc:.4f}")
    lines.append(f"    MAE CAF        : {mae_caf:.4f}")
    lines.append(f"    Delta CAF-CONMEBOL: {mae_caf - mae_conmebol:+.4f}")
    lines.append(f"    Ratio imputado/real: {ratio_d2:.3f}")
    lines.append(f"    r(win_rate, gls): {r_wg:.3f}")
    lines.append(f"    Var/Média global: {overall_disp:.3f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 70
    header = [
        sep,
        "  DIAGNÓSTICO DO MODELO POISSON — COPA 2026",
        f"  Gerado em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Dataset   : {INPUT_CSV}",
        f"  Cutoff    : {CUTOFF}  (treino ≤ | teste >)",
        sep,
    ]

    print("\n".join(header))
    print("Carregando dados...")

    wide = load_data()
    long = wide_to_long(wide)

    train_l = long[long["date"] <= CUTOFF]
    print(f"  Treino: {len(wide[wide['date'] <= CUTOFF])} jogos | "
          f"Teste: {len(wide[wide['date'] > CUTOFF])} jogos")

    print("Gerando predições no conjunto de teste...")
    pred_df, _ = get_test_predictions(wide, long)
    print(f"  {len(pred_df)} jogos no conjunto de teste com predições\n")

    # Roda os 4 diagnósticos
    d1 = diag1_mae_by_conf(pred_df)
    d2 = diag2_imputation_impact(pred_df)
    d3 = diag3_correlation(train_l)
    d4 = diag4_overdispersion(train_l)
    c  = conclusao(pred_df, train_l)

    full_report = "\n".join(header) + "\n" + d1 + d2 + d3 + d4 + c + "\n"

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(full_report)
    print(f"\nRelatório salvo em {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
