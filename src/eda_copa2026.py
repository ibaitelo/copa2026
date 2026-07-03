"""
EDA Copa 2026 — 4 perguntas estratégicas antes do build_features.py
Fonte: data/processed/all_confederations_clean.csv
       data/raw/friendlies_raw.csv
Saídas: outputs/eda_report.txt
        outputs/feature_coverage_map.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
CLEAN_CSV      = Path("data/processed/all_confederations_clean.csv")
FRIENDLIES_CSV = Path("data/raw/friendlies_raw.csv")
OUT_DIR        = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH    = OUT_DIR / "eda_report.txt"
COVERAGE_PATH  = OUT_DIR / "feature_coverage_map.csv"

# ---------------------------------------------------------------------------
# Times Copa 2026 confirmados (46 — 2 playoffs ainda sem confirmação)
# ---------------------------------------------------------------------------
COPA2026: list[str] = [
    # CONMEBOL (6)
    "Argentina", "Brazil", "Colombia", "Uruguay", "Ecuador", "Venezuela",
    # UEFA (16)
    "Spain", "Germany", "France", "England", "Portugal", "Netherlands",
    "Italy", "Türkiye", "Austria", "Scotland", "Switzerland", "Croatia",
    "Serbia", "Hungary", "Slovenia", "Romania",
    # AFC (8)
    "Japan", "Korea Republic", "IR Iran", "Iraq", "Jordan", "Australia",
    "Uzbekistan", "Saudi Arabia",
    # CONCACAF (6)
    "United States", "Mexico", "Canada", "Panama", "Honduras", "Jamaica",
    # CAF (9)
    "Morocco", "Senegal", "Egypt", "Nigeria", "Cameroon", "Côte d'Ivoire",
    "South Africa", "Tunisia", "Congo DR",
    # OFC (1)
    "New Zealand",
]

COPA_SET = set(COPA2026)

STAT_FEATURES = {
    "gls":  "performance_gls",
    "ast":  "performance_ast",
    "sh":   "performance_sh",
    "sot":  "performance_sot",
    "tklw": "performance_tklw",
    "int":  "performance_int",
}

# ---------------------------------------------------------------------------
# Buffer de relatório
# ---------------------------------------------------------------------------
_lines: list[str] = []

def _w(*args):
    line = " ".join(str(a) for a in args)
    _lines.append(line)

def _sec(title: str):
    _w()
    _w("=" * 68)
    _w(f"  {title}")
    _w("=" * 68)

def _sub(title: str):
    _w()
    _w(f"  -- {title} --")


# ---------------------------------------------------------------------------
# Helpers de agregação
# ---------------------------------------------------------------------------

def games_per_team(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame de jogos únicos por team_name (dedupicando por match_id)."""
    return (
        df.drop_duplicates(subset=["match_id", "team_name"])
        [["match_id", "team_name", "confederation", "has_shot_data"]]
    )


def team_goals_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Soma gls por (match_id, team_name) → gols marcados por time por jogo."""
    if "performance_gls" not in df.columns:
        return pd.DataFrame(columns=["match_id", "team_name", "confederation", "gls_team"])
    agg = (
        df.groupby(["match_id", "team_name", "confederation"], dropna=False)["performance_gls"]
        .sum(min_count=1)
        .reset_index(name="gls_team")
    )
    return agg.dropna(subset=["gls_team"])


# ---------------------------------------------------------------------------
# PERGUNTA 1 — Volume por time da Copa 2026
# ---------------------------------------------------------------------------

def pergunta1(df: pd.DataFrame) -> list[str]:
    _sec("PERGUNTA 1 — Volume por time da Copa 2026")
    alerts: list[str] = []

    gpt = games_per_team(df)
    copa_games = gpt[gpt["team_name"].isin(COPA_SET)]

    volume = (
        copa_games.groupby("team_name")
        .agg(
            total_jogos   = ("match_id", "nunique"),
            com_shot      = ("has_shot_data", "sum"),
        )
        .reset_index()
    )
    volume["sem_shot"] = volume["total_jogos"] - volume["com_shot"]
    volume = volume.sort_values("total_jogos", ascending=False).reset_index(drop=True)

    # Adiciona confederação para contexto
    conf_map = (
        copa_games.drop_duplicates("team_name")
        .set_index("team_name")["confederation"]
    )
    volume["conf"] = volume["team_name"].map(conf_map)

    _sub("Todos os 46 times — jogos por time")
    _w(f"\n  {'Time':<22} {'Conf':<10} {'Total':>6} {'Com sh':>7} {'Sem sh':>7}")
    _w("  " + "-" * 55)
    for _, r in volume.iterrows():
        marker = " ⚠" if r["total_jogos"] < 6 else ""
        _w(f"  {r['team_name']:<22} {str(r['conf']):<10} {r['total_jogos']:>6} "
           f"{r['com_shot']:>7} {r['sem_shot']:>7}{marker}")

    below6 = volume[volume["total_jogos"] < 6]
    _sub(f"Times com menos de 6 jogos no total ({len(below6)})")
    if len(below6) > 0:
        for _, r in below6.iterrows():
            _w(f"  {r['team_name']:<22}  {r['total_jogos']} jogos")
            alerts.append(f"[P1] '{r['team_name']}': apenas {r['total_jogos']} jogos — janela móvel incompleta")
    else:
        _w("  Todos os 46 times têm ≥ 6 jogos.")

    _w(f"\n  Mediana jogos por time: {volume['total_jogos'].median():.0f}")
    _w(f"  Mínimo: {volume['total_jogos'].min()} ({volume.loc[volume['total_jogos'].idxmin(),'team_name']})")
    _w(f"  Máximo: {volume['total_jogos'].max()} ({volume.loc[volume['total_jogos'].idxmax(),'team_name']})")

    return alerts


# ---------------------------------------------------------------------------
# PERGUNTA 2 — Cobertura dos amistosos 2026
# ---------------------------------------------------------------------------

def pergunta2(df_clean: pd.DataFrame) -> list[str]:
    _sec("PERGUNTA 2 — Cobertura dos amistosos 2026")
    alerts: list[str] = []

    if not FRIENDLIES_CSV.exists():
        _w("  ERRO: friendlies_raw.csv não encontrado.")
        return alerts

    fr = pd.read_csv(FRIENDLIES_CSV, low_memory=False)
    if "date" in fr.columns:
        fr["date"] = pd.to_datetime(fr["date"], errors="coerce")
    if "year" not in fr.columns and "date" in fr.columns:
        fr["year"] = fr["date"].dt.year

    fr_2026 = fr[fr["year"] == 2026].copy()
    if len(fr_2026) == 0:
        _w("  Nenhum amistoso encontrado para 2026.")
        return alerts

    gpt_fr = fr_2026.drop_duplicates(subset=["match_id", "team_name"])[
        ["match_id", "team_name", "date"]
    ]

    copa_fr = gpt_fr[gpt_fr["team_name"].isin(COPA_SET)]
    team_counts = copa_fr.groupby("team_name")["match_id"].nunique().sort_values(ascending=False)

    _sub("Amistosos 2026 — times da Copa com dados")
    covered = set(team_counts.index)
    not_covered = COPA_SET - covered
    _w(f"\n  Times da Copa com amistosos 2026: {len(covered)}/{len(COPA2026)}")
    _w(f"  Times SEM amistosos 2026: {len(not_covered)}")

    _w(f"\n  {'Time':<22} {'Amistosos 2026':>15}")
    _w("  " + "-" * 38)
    for team, cnt in team_counts.items():
        _w(f"  {team:<22} {cnt:>15}")

    if not_covered:
        _sub("Times da Copa SEM amistosos registrados em 2026")
        for t in sorted(not_covered):
            _w(f"  - {t}")
            alerts.append(f"[P2] '{t}': sem amistosos 2026 — features pré-Copa menos frescas")

    # Data mais recente por time (jan–jun 2026)
    _sub("Data do amistoso mais recente (jan–jun 2026)")
    last_date = copa_fr.groupby("team_name")["date"].max().sort_values(ascending=False)
    for team, dt in last_date.items():
        _w(f"  {team:<22}  {str(dt)[:10]}")

    return alerts


# ---------------------------------------------------------------------------
# PERGUNTA 3 — Distribuição de gols por confederação
# ---------------------------------------------------------------------------

def pergunta3(df: pd.DataFrame) -> list[str]:
    _sec("PERGUNTA 3 — Distribuição de gols por confederação")
    alerts: list[str] = []

    copa_df = df[df["team_name"].isin(COPA_SET)].copy()
    goals = team_goals_per_game(copa_df)
    if goals.empty:
        _w("  ERRO: dados de gols ausentes.")
        return alerts

    _sub("Média de gols marcados por jogo (por confederação)")
    conf_stats = (
        goals.groupby("confederation")["gls_team"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .round(3)
    )
    _w(conf_stats.to_string())

    # Teste estatístico: Kruskal-Wallis (não paramétrico, mais robusto)
    _sub("Teste Kruskal-Wallis (H0: distribuições iguais entre confederações)")
    groups = [grp["gls_team"].values for _, grp in goals.groupby("confederation") if len(grp) >= 5]
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        _w(f"  H = {h_stat:.4f},  p = {p_val:.6f}")
        if p_val < 0.05:
            _w(f"  → p < 0.05: distribuições DIFERENTES (p={p_val:.4f})")
            _w("  → Recomendação: incluir intercepto por confederação no modelo.")
            alerts.append(f"[P3] Kruskal-Wallis p={p_val:.4f}: médias de gols diferem por confederação")
        else:
            _w(f"  → p ≥ 0.05: sem evidência de diferença significativa (p={p_val:.4f})")
    else:
        _w("  Grupos insuficientes para teste.")

    # Post-hoc: comparação par a par
    _sub("Médias de gols por confederação (ordenado)")
    ranking = conf_stats["mean"].sort_values(ascending=False)
    for conf, mean in ranking.items():
        n = int(conf_stats.loc[conf, "n"])
        _w(f"  {conf:<12}  média={mean:.3f}  n={n}")

    return alerts


# ---------------------------------------------------------------------------
# PERGUNTA 4 — Mapa de features disponíveis por time
# ---------------------------------------------------------------------------

def pergunta4(df: pd.DataFrame) -> list[str]:
    _sec("PERGUNTA 4 — Mapa de features disponíveis por time Copa 2026")
    alerts: list[str] = []

    copa_df = df[df["team_name"].isin(COPA_SET)].copy()

    # Para cada time + feature: conta match_ids onde há pelo menos 1 valor não-nulo
    rows = []
    for team in COPA2026:
        t_df = copa_df[copa_df["team_name"] == team]
        total_games = t_df["match_id"].nunique()
        row = {"time": team, "total_jogos": total_games}
        for feat_short, feat_col in STAT_FEATURES.items():
            if feat_col in t_df.columns:
                # Conta jogos com pelo menos um jogador com o dado
                games_with = (
                    t_df.dropna(subset=[feat_col])
                    .groupby("match_id")[feat_col]
                    .count()
                    .gt(0)
                    .sum()
                )
                row[feat_short] = int(games_with)
            else:
                row[feat_short] = 0
        rows.append(row)

    coverage = pd.DataFrame(rows)
    coverage.to_csv(COVERAGE_PATH, index=False)
    _w(f"  Salvo em: {COVERAGE_PATH}")

    _sub("10 times com pior cobertura (por jogos com 'sh' disponível)")
    worst = coverage.nsmallest(10, "sh")[
        ["time", "total_jogos"] + list(STAT_FEATURES.keys())
    ]
    _w(f"\n  {'Time':<22} {'Total':>6} {'gls':>5} {'ast':>5} {'sh':>5} {'sot':>5} {'tklw':>5} {'int':>5}")
    _w("  " + "-" * 62)
    for _, r in worst.iterrows():
        _w(f"  {r['time']:<22} {r['total_jogos']:>6} "
           f"{r['gls']:>5} {r['ast']:>5} {r['sh']:>5} "
           f"{r['sot']:>5} {r['tklw']:>5} {r['int']:>5}")
        if r["sh"] < 3:
            alerts.append(f"[P4] '{r['time']}': apenas {int(r['sh'])} jogos com sh — janela móvel muito limitada")

    _sub("10 times com melhor cobertura (por 'sh')")
    best = coverage.nlargest(10, "sh")[
        ["time", "total_jogos"] + list(STAT_FEATURES.keys())
    ]
    _w(f"\n  {'Time':<22} {'Total':>6} {'gls':>5} {'ast':>5} {'sh':>5} {'sot':>5} {'tklw':>5} {'int':>5}")
    _w("  " + "-" * 62)
    for _, r in best.iterrows():
        _w(f"  {r['time']:<22} {r['total_jogos']:>6} "
           f"{r['gls']:>5} {r['ast']:>5} {r['sh']:>5} "
           f"{r['sot']:>5} {r['tklw']:>5} {r['int']:>5}")

    return alerts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 68)
    print("  EDA COPA 2026")
    print("=" * 68)

    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print(f"  Dataset: {len(df):,} linhas, {df['match_id'].nunique()} jogos, "
          f"{df['team_name'].nunique()} times")

    alerts1 = pergunta1(df)
    alerts2 = pergunta2(df)
    alerts3 = pergunta3(df)
    alerts4 = pergunta4(df)

    all_alerts = alerts1 + alerts2 + alerts3 + alerts4

    # Resumo executivo
    _sec("RESUMO EXECUTIVO")
    _w(f"  Dataset: {len(df):,} linhas | {df['match_id'].nunique()} jogos | "
       f"{df['team_name'].nunique()} times únicos")
    _w(f"  Times Copa 2026 confirmados: {len(COPA2026)}")
    _w()
    if all_alerts:
        _w(f"  ALERTAS ({len(all_alerts)}):")
        for a in all_alerts:
            _w(f"    ⚠  {a}")
    else:
        _w("  Nenhum alerta crítico.")

    # Salva relatório
    REPORT_PATH.write_text("\n".join(_lines), encoding="utf-8")

    # Imprime resumo no terminal
    print()
    print("=" * 68)
    print("  RESUMO EXECUTIVO")
    print("=" * 68)
    print(f"  {len(all_alerts)} alertas identificados:")
    for a in all_alerts:
        print(f"    ⚠  {a}")
    print()
    print(f"  Relatório completo: {REPORT_PATH}")
    print(f"  Coverage map:       {COVERAGE_PATH}")
    print()


if __name__ == "__main__":
    main()
