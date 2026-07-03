#!/usr/bin/env python3
"""
feature_quality_v12.py — Diagnóstico completo de qualidade das features v12

Verifica:
1. Cobertura (missing rate) por feature e por time
2. Outliers (IQR 3× e z-score > 4)
3. Correlações entre features (colinearidade)
4. SHAP values — importância global e por confederação
5. Calibração do modelo (Poisson deviance e MAE por confederação)
6. Incongruências: times sem Copa 2026, saves=0 suspeito, features invertidas
7. Sumário com recomendações

Output: feature_quality_v12.json (para consumo por retrain script)
        feature_quality_report_v12.txt (leitura humana)
"""
from __future__ import annotations
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PKL   = Path("outputs/xgboost_v12.pkl")
MODEL_CSV   = Path("data/processed/model_dataset.csv")
RAW_CSV     = Path("data/raw/full_dataset_raw.csv")
ELO_CSV     = Path("data/raw/elo_history.csv")
OUT_JSON    = Path("outputs/feature_quality_v12.json")
OUT_TXT     = Path("outputs/feature_quality_report_v12.txt")

CONF_DUMMIES = ["conf_AFC","conf_CAF","conf_CONCACAF","conf_CONMEBOL","conf_OFC","conf_UEFA"]
RATING_FEATURES = [
    "elo_diff_decay","delta_rating_decay","gls_ponderado_decay",
    "gls_decay_vs_forte","gls_decay_vs_fraco","margem_gols_decay",
]
FEATURES = (["opp_saves_decay","shots_on_goal_decay","gols_sofridos_decay"]
            + RATING_FEATURES + CONF_DUMMIES)

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

ALL_COPA_TEAMS = list(COPA_TEAM_CONF.keys())
CUTOFF = pd.Timestamp("2026-06-28")

SEP = "=" * 72
lines: list[str] = []
issues: list[dict] = []

def p(msg=""):
    print(msg); lines.append(msg)

def warn(severity: str, code: str, team: str, msg: str, value=None):
    issues.append({"severity": severity, "code": code, "team": team, "msg": msg, "value": value})
    label = {"CRITICAL":"❌","HIGH":"⚠️ ","MEDIUM":"⚡","INFO":"ℹ️ "}.get(severity, "?")
    p(f"  {label} [{severity}] {team}: {msg}" + (f" (val={value:.3f})" if isinstance(value, float) else ""))


# =============================================================================
# 1. Carregar dados
# =============================================================================
p(SEP); p("  SEÇÃO 1 — Carregando dados"); p(SEP)

wide = pd.read_csv(MODEL_CSV, parse_dates=["date"])
raw  = pd.read_csv(RAW_CSV, parse_dates=["date"])
with open(MODEL_PKL, "rb") as f:
    model = pickle.load(f)

p(f"  model_dataset: {len(wide):,} jogos wide")
p(f"  raw: {len(raw):,} linhas, {raw['match_id'].nunique()} jogos")
p(f"  Copa 2026 (WC) no raw: {(raw['match_type']=='WC').sum()//2} jogos")

# Construir long format com features para análise
rows = []
for _, r in wide.iterrows():
    for side, opp in [("home","away"),("away","home")]:
        row = {"date": r["date"], "team": r[f"{side}_team"]}
        for feat in FEATURES:
            if feat.startswith("conf_"):
                row[feat] = int(r.get(feat, 0))
            else:
                v = r.get(f"{side}_{feat}", np.nan)
                row[feat] = float(v) if pd.notna(v) else np.nan
        row["gols_marcados"] = r[f"{side}_gols"]
        row["match_type"] = r.get("match_type", "")
        rows.append(row)
long = pd.DataFrame(rows)

train = long[long["date"] <= CUTOFF].copy()
p(f"  Training set: {len(train):,} obs")


# =============================================================================
# 2. Cobertura de features por time (Copa 2026)
# =============================================================================
p(); p(SEP); p("  SEÇÃO 2 — Cobertura de features por time (Copa 2026)"); p(SEP)

copa_long = long[long["date"] <= CUTOFF].copy()
copa_long["is_copa"] = long["date"] >= pd.Timestamp("2026-06-11")

# Features numéricas (excluindo conf_*)
num_feats = ["opp_saves_decay","shots_on_goal_decay","gols_sofridos_decay"] + RATING_FEATURES

coverage: dict[str, dict] = {}
for team in ALL_COPA_TEAMS:
    td = copa_long[copa_long["team"] == team]
    if len(td) == 0:
        warn("HIGH", "NO_DATA", team, "Time sem nenhuma observação no dataset")
        coverage[team] = {}
        continue

    last_obs = td["date"].max()
    n_total  = len(td)
    n_copa   = int((td["date"] >= pd.Timestamp("2026-06-11")).sum())

    team_cov: dict[str, float] = {}
    for feat in num_feats:
        if feat not in td.columns:
            team_cov[feat] = 0.0; continue
        rate = float(td[feat].notna().mean())
        team_cov[feat] = rate

    coverage[team] = {
        "n_total": n_total, "n_copa": n_copa,
        "last_obs": str(last_obs.date()),
        "feature_coverage": team_cov,
    }

    # Avisos por cobertura baixa
    for feat, rate in team_cov.items():
        if rate < 0.3:
            warn("HIGH", "LOW_COVERAGE", team, f"{feat} cobertura baixa ({rate:.0%})", rate)
        elif rate < 0.6:
            warn("MEDIUM", "MED_COVERAGE", team, f"{feat} cobertura média ({rate:.0%})", rate)

# Resumo de cobertura
p("\n  Cobertura média por feature (todos os times Copa):")
for feat in num_feats:
    rates = [d["feature_coverage"].get(feat, 0) for d in coverage.values() if "feature_coverage" in d]
    p(f"  {'  ' if len(feat)<25 else ''}{feat:<30} {np.mean(rates):5.1%}  (min={np.min(rates):.0%})")

# Times sem dados Copa 2026
teams_no_copa = [t for t in ALL_COPA_TEAMS if coverage.get(t, {}).get("n_copa", 0) == 0]
if teams_no_copa:
    p(f"\n  Times SEM jogos Copa 2026 no dataset: {teams_no_copa}")
    for t in teams_no_copa:
        warn("HIGH", "NO_COPA_GAMES", t, "Sem jogos Copa 2026 no dataset (features desatualizadas)")
else:
    p("\n  ✓ Todos os 48 times têm jogos Copa 2026 no dataset")


# =============================================================================
# 3. Outliers
# =============================================================================
p(); p(SEP); p("  SEÇÃO 3 — Outliers"); p(SEP)

outlier_summary: dict = {}
for feat in num_feats:
    if feat not in train.columns: continue
    col = train[feat].dropna()
    if len(col) < 10: continue
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 3*iqr, q3 + 3*iqr
    z = (col - col.mean()) / (col.std() + 1e-9)
    n_iqr = int(((col < lo) | (col > hi)).sum())
    n_z4  = int((z.abs() > 4).sum())
    outlier_summary[feat] = {"n_iqr3": n_iqr, "n_z4": n_z4,
                              "mean": float(col.mean()), "std": float(col.std()),
                              "min": float(col.min()), "max": float(col.max()),
                              "p1": float(col.quantile(0.01)), "p99": float(col.quantile(0.99))}
    status = "✓" if n_iqr == 0 else ("⚠" if n_iqr < 10 else "❌")
    p(f"  {status} {feat:<30} mean={col.mean():6.3f}  std={col.std():5.3f}  "
      f"[{col.min():.2f}, {col.max():.2f}]  outliers_IQR3={n_iqr}  z>4={n_z4}")
    if n_iqr > 20:
        warn("HIGH", "MANY_OUTLIERS", feat, f"Muitos outliers IQR3: {n_iqr} obs", float(n_iqr))
    elif n_iqr > 5:
        warn("MEDIUM", "SOME_OUTLIERS", feat, f"Alguns outliers IQR3: {n_iqr} obs", float(n_iqr))


# =============================================================================
# 4. Correlações (colinearidade)
# =============================================================================
p(); p(SEP); p("  SEÇÃO 4 — Correlações entre features"); p(SEP)

corr_mat = train[num_feats].corr()
p("  Pares com |r| > 0.70 (colinearidade alta):")
pairs_found = 0
corr_issues: list[dict] = []
for i, f1 in enumerate(num_feats):
    for f2 in num_feats[i+1:]:
        r = corr_mat.loc[f1, f2]
        if abs(r) > 0.70:
            severity = "HIGH" if abs(r) > 0.90 else "MEDIUM"
            p(f"  {'❌' if abs(r)>0.90 else '⚠️ '} {f1:<28} vs {f2:<28}  r={r:.3f}")
            warn(severity, "HIGH_CORR", f"{f1}↔{f2}",
                 f"Correlação alta ({r:.2f}) — risco de colinearidade", r)
            corr_issues.append({"f1": f1, "f2": f2, "r": round(r,4)})
            pairs_found += 1
if pairs_found == 0:
    p("  ✓ Nenhum par com correlação > 0.70")


# =============================================================================
# 5. SHAP — Importância das features
# =============================================================================
p(); p(SEP); p("  SEÇÃO 5 — SHAP Importância das features"); p(SEP)

try:
    import shap
    X_train = train[FEATURES].fillna(train[FEATURES].median())
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
    mean_abs_shap = mean_abs_shap.sort_values(ascending=False)

    p("  Feature importance (SHAP mean |value|):")
    shap_dict: dict[str, float] = {}
    for feat, val in mean_abs_shap.items():
        bar = "█" * int(val * 30 / mean_abs_shap.iloc[0])
        p(f"    {feat:<30} {val:.4f}  {bar}")
        shap_dict[feat] = round(float(val), 6)

    # Features com SHAP muito baixo (quase irrelevantes)
    threshold = mean_abs_shap.mean() * 0.15
    low_shap = mean_abs_shap[mean_abs_shap < threshold]
    if not low_shap.empty:
        p(f"\n  Features com SHAP < {threshold:.4f} (potencialmente removíveis):")
        for feat, val in low_shap.items():
            p(f"    {feat}: {val:.4f}")
            warn("INFO", "LOW_SHAP", feat, f"Importância SHAP muito baixa ({val:.4f})", val)

    # Check se conf_* têm SHAP consistente com tamanho dos grupos
    conf_shaps = {f: shap_dict.get(f, 0) for f in CONF_DUMMIES}
    p(f"\n  SHAP confederações: {conf_shaps}")
    shap_ok = True
except ImportError:
    p("  ⚠️  SHAP não instalado — pulando análise SHAP")
    shap_dict = {}
    shap_ok = False


# =============================================================================
# 6. Calibração do modelo
# =============================================================================
p(); p(SEP); p("  SEÇÃO 6 — Calibração do modelo"); p(SEP)

X_tr = train[FEATURES].fillna(train[FEATURES].median())
y_tr = train["gols_marcados"].fillna(0)
y_pred = model.predict(X_tr)

mae_global  = float(np.mean(np.abs(y_pred - y_tr)))
rmse_global = float(np.sqrt(np.mean((y_pred - y_tr)**2)))
# Poisson deviance
eps = 1e-9
pd_dev = float(2 * np.mean(y_pred - y_tr - y_tr * np.log((y_pred + eps)/(y_tr + eps))))

p(f"  MAE global:           {mae_global:.4f}")
p(f"  RMSE global:          {rmse_global:.4f}")
p(f"  Poisson deviance:     {pd_dev:.4f}")

# Por confederação
p("\n  MAE por confederação (última janela):")
calibration: dict[str, dict] = {}
for conf in CONF_DUMMIES:
    conf_name = conf.replace("conf_","")
    mask = train[conf].astype(bool)
    if mask.sum() < 10: continue
    mae_c = float(np.mean(np.abs(y_pred[mask] - y_tr.values[mask])))
    mean_pred = float(y_pred[mask].mean())
    mean_real = float(y_tr.values[mask].mean())
    bias = mean_pred - mean_real
    p(f"  {conf_name:<10} n={mask.sum():>4}  MAE={mae_c:.3f}  "
      f"pred_mean={mean_pred:.2f}  real_mean={mean_real:.2f}  bias={bias:+.3f}")
    calibration[conf_name] = {"mae": mae_c, "mean_pred": mean_pred,
                               "mean_real": mean_real, "bias": bias, "n": int(mask.sum())}
    if abs(bias) > 0.35:
        warn("HIGH", "CALIBRATION_BIAS", conf_name,
             f"Viés sistemático alto: pred={mean_pred:.2f} vs real={mean_real:.2f}", bias)
    elif abs(bias) > 0.15:
        warn("MEDIUM", "CALIBRATION_BIAS", conf_name,
             f"Viés moderado: pred={mean_pred:.2f} vs real={mean_real:.2f}", bias)

# Copa 2026 games calibration
wc_mask = train["match_type"] == "WC"
if wc_mask.sum() > 0:
    mae_wc   = float(np.mean(np.abs(y_pred[wc_mask] - y_tr.values[wc_mask])))
    pred_wc  = float(y_pred[wc_mask].mean())
    real_wc  = float(y_tr.values[wc_mask].mean())
    p(f"\n  Copa 2026 (WC) n={wc_mask.sum()}  MAE={mae_wc:.3f}  "
      f"pred_mean={pred_wc:.2f}  real_mean={real_wc:.2f}  bias={pred_wc-real_wc:+.3f}")
    if abs(pred_wc - real_wc) > 0.25:
        warn("HIGH", "WC_CALIBRATION", "Copa2026",
             f"Viés no ajuste Copa 2026: pred={pred_wc:.2f} real={real_wc:.2f}", pred_wc - real_wc)


# =============================================================================
# 7. Incongruências específicas
# =============================================================================
p(); p(SEP); p("  SEÇÃO 7 — Incongruências e pontos de consideração"); p(SEP)

# 7a. Saves = 0 suspeito (time sem goleiro?)
saves_zero = train[(train["opp_saves_decay"] < 0.05) & train["opp_saves_decay"].notna()]
if not saves_zero.empty:
    teams_zero = saves_zero["team"].value_counts().head(10)
    p("  Times com opp_saves_decay ≈ 0 (suspeito — goleiro sem dados?):")
    for t, cnt in teams_zero.items():
        p(f"    {t}: {cnt} obs com saves≈0")
        warn("MEDIUM", "SAVES_ZERO", t, f"opp_saves_decay≈0 em {cnt} obs", float(cnt))

# 7b. Lambda invertida: time "fraco" com λ muito alto
p("\n  Verificando coerência lambda para times com ELO extremo...")
if ELO_CSV.exists():
    elo_df = pd.read_csv(ELO_CSV, parse_dates=["date"])
    elo_last = elo_df.sort_values("date").groupby("team")["elo_after"].last()

    copa_elos = {t: float(elo_last.get(t, 1500)) for t in ALL_COPA_TEAMS}
    elo_sorted = sorted(copa_elos.items(), key=lambda x: x[1], reverse=True)
    p(f"  Top-5 ELO Copa: {[(t, int(e)) for t, e in elo_sorted[:5]]}")
    p(f"  Bot-5 ELO Copa: {[(t, int(e)) for t, e in elo_sorted[-5:]]}")

    # Check: times muito fracos não devem ter opp_saves alto (nunca finalizaram)
    for team in [t for t, _ in elo_sorted[-10:]]:
        td = train[train["team"] == team]
        if len(td) == 0: continue
        avg_sot = td["shots_on_goal_decay"].mean()
        if avg_sot > 5.0:
            warn("MEDIUM", "SOT_INCONSISTENT", team,
                 f"Time fraco (ELO={copa_elos[team]:.0f}) mas shots_on_goal_decay alto={avg_sot:.2f}", avg_sot)

# 7c. Copa 2026 — features atualizam corretamente?
p("\n  Verificando se Copa 2026 atualizou features do time:")
copa_raw = raw[raw["match_type"] == "WC"].copy()
copa_raw["date"] = pd.to_datetime(copa_raw["date"])
teams_no_update = []
for team in ALL_COPA_TEAMS:
    team_copa = copa_raw[copa_raw["team"] == team]
    if len(team_copa) == 0:
        teams_no_update.append(team)
if teams_no_update:
    p(f"  Times sem dados WC no raw: {teams_no_update}")
    for t in teams_no_update:
        warn("CRITICAL", "NO_WC_RAW", t, "Time sem jogos WC no raw dataset")
else:
    p("  ✓ Todos os 48 times têm jogos WC no raw dataset")

# 7d. Verifica se home_advantage está zerado
ha_nonzero = int((raw["home_advantage"] != 0).sum())
p(f"\n  home_advantage != 0 no raw: {ha_nonzero} linhas (esperado: 0)")
if ha_nonzero > 0:
    warn("CRITICAL", "HOME_ADV_NONZERO", "dataset",
         f"home_advantage != 0 ainda presente em {ha_nonzero} linhas!", float(ha_nonzero))
else:
    p("  ✓ home_advantage = 0 em todas as linhas")

# 7e. Check feature: gls_decay_vs_forte vs gls_decay_vs_fraco invertidos?
ratio_forte = train["gls_decay_vs_forte"].dropna()
ratio_fraco = train["gls_decay_vs_fraco"].dropna()
if len(ratio_forte) > 100 and len(ratio_fraco) > 100:
    m_forte = ratio_forte.mean()
    m_fraco = ratio_fraco.mean()
    p(f"\n  gls_decay_vs_forte média: {m_forte:.3f}")
    p(f"  gls_decay_vs_fraco média: {m_fraco:.3f}")
    if m_forte > m_fraco + 0.2:
        warn("HIGH", "FEATURE_INVERTED", "gls_decay_vs_forte",
             f"Média vs forte ({m_forte:.2f}) > vs fraco ({m_fraco:.2f}) — semanticamente invertido!", m_forte - m_fraco)
        p("  ⚠️  ATENÇÃO: times marcam MAIS contra fortes que contra fracos — incoerente!")
        p("       Possível causa: ELO_FORTE_THRESHOLD (1750) muito alto, poucos jogos vs 'forte'")
    else:
        p(f"  ✓ Coerente: vs_forte={m_forte:.2f} < vs_fraco={m_fraco:.2f}")

# 7f. Teams with suspicious lambda (too low or too high)
p("\n  Verificando times sem shots_on_goal_decay (feature mais importante):")
sot_coverage = {t: float(train[train["team"]==t]["shots_on_goal_decay"].notna().mean())
                for t in ALL_COPA_TEAMS if len(train[train["team"]==t]) > 0}
low_sot = {t: v for t, v in sot_coverage.items() if v < 0.50}
if low_sot:
    p(f"  Times com shots_on_goal_decay < 50%: {low_sot}")
    for t, v in low_sot.items():
        warn("HIGH", "LOW_SOT_COV", t, f"shots_on_goal_decay cobertura={v:.0%}", v)
else:
    p("  ✓ Todos os times têm shots_on_goal_decay > 50%")


# =============================================================================
# 8. Sumário de issues e recomendações
# =============================================================================
p(); p(SEP); p("  SEÇÃO 8 — SUMÁRIO E RECOMENDAÇÕES"); p(SEP)

by_sev = {}
for iss in issues:
    by_sev.setdefault(iss["severity"], []).append(iss)

total = len(issues)
p(f"  Total de issues encontradas: {total}")
for sev in ["CRITICAL","HIGH","MEDIUM","INFO"]:
    n = len(by_sev.get(sev, []))
    label = {"CRITICAL":"❌","HIGH":"⚠️ ","MEDIUM":"⚡","INFO":"ℹ️ "}.get(sev, "?")
    if n > 0:
        p(f"  {label} {sev}: {n}")

p("\n  Recomendações prioritárias:")

# Agrupar por código
by_code: dict[str, list] = {}
for iss in issues:
    by_code.setdefault(iss["code"], []).append(iss)

recs = []
if "NO_COPA_GAMES" in by_code:
    recs.append(("CRITICAL", "Times sem jogos Copa 2026: features desatualizadas. "
                  "Verificar se todos os jogos WC foram corretamente importados."))
if "HOME_ADV_NONZERO" in by_code:
    recs.append(("CRITICAL", "home_advantage != 0 ainda presente. Re-executar correção."))
if "WC_CALIBRATION" in by_code:
    bias = by_code["WC_CALIBRATION"][0].get("value", 0)
    recs.append(("HIGH", f"Modelo superestima/subestima gols Copa 2026 em {bias:+.2f}. "
                  "Considerar recalibração com scala pós-predição."))
if "FEATURE_INVERTED" in by_code:
    recs.append(("HIGH", "Feature gls_decay_vs_forte > gls_decay_vs_fraco — semanticamente invertida. "
                  "ELO_FORTE_THRESHOLD=1750 pode ser muito alto para Copa 2026 (times mais equilibrados). "
                  "Recalibrar threshold ou remover e usar gls_ponderado_decay."))
if "HIGH_CORR" in by_code:
    pairs = [(d["team"], d["value"]) for d in by_code["HIGH_CORR"]]
    recs.append(("MEDIUM", f"Colinearidade alta detectada: {pairs[:3]}. "
                  "XGBoost é robusto, mas pode causar instabilidade em importâncias."))
if "CALIBRATION_BIAS" in by_code:
    biased = [d for d in by_code["CALIBRATION_BIAS"] if d["severity"]=="HIGH"]
    if biased:
        recs.append(("HIGH", f"Viés sistemático em confederações: "
                      f"{[d['team'] for d in biased]}. Verificar normalização de saves por conf."))
if "MANY_OUTLIERS" in by_code:
    recs.append(("MEDIUM", "Outliers excessivos detectados. Considerar winsorização em p1/p99."))

for sev, rec in recs:
    label = {"CRITICAL":"❌","HIGH":"⚠️ ","MEDIUM":"⚡"}.get(sev, "ℹ")
    p(f"  {label} {rec}")
    p()

# =============================================================================
# 9. Salvar outputs
# =============================================================================
report = "\n".join(lines)
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text(report, encoding="utf-8")
p(f"\n  Relatório salvo: {OUT_TXT}")

quality_data = {
    "issues": issues,
    "by_code": {k: len(v) for k, v in by_code.items()},
    "outliers": outlier_summary,
    "calibration": calibration,
    "shap": shap_dict,
    "corr_issues": corr_issues,
    "coverage": {t: {k: v for k, v in d.items() if k != "feature_coverage"}
                  for t, d in coverage.items()},
    "mae_global": mae_global,
    "rmse_global": rmse_global,
    "poisson_deviance": pd_dev,
}
OUT_JSON.write_text(json.dumps(quality_data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
p(f"  JSON salvo: {OUT_JSON}")


if __name__ == "__main__":
    pass  # all code runs at module level for output capture
