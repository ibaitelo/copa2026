"""
data_quality_report_48.py — Relatório completo de qualidade de dados para as 48 seleções da Copa 2026.

Fontes consolidadas:
  - data/raw/api_football_wcq_raw.csv   (WCQ das 6 confederações)
  - data/raw/hosts_competitive_raw.csv  (Copa América, CONCACAF NL, Gold Cup — países-sede)

Saída: outputs/data_quality_report_48.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_PATH = Path("outputs/data_quality_report_48.txt")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 48 seleções classificadas (nome normalizado → nome na API)
# ---------------------------------------------------------------------------
QUALIFIED_48 = {
    "UEFA": [
        "Austria", "Belgium", "Bosnia and Herzegovina", "Croatia", "Czech Republic",
        "England", "France", "Germany", "Netherlands", "Norway", "Portugal",
        "Scotland", "Spain", "Sweden", "Switzerland", "Turkey",
    ],
    "CAF": [
        "Algeria", "Cape Verde", "DR Congo", "Egypt", "Ghana", "Ivory Coast",
        "Morocco", "Senegal", "South Africa", "Tunisia",
    ],
    "AFC": [
        "Australia", "Iran", "Iraq", "Japan", "Jordan", "Qatar",
        "Saudi Arabia", "South Korea", "Uzbekistan",
    ],
    "CONCACAF": ["Canada", "Curacao", "Haiti", "Mexico", "Panama", "United States"],
    "CONMEBOL": ["Argentina", "Brazil", "Colombia", "Ecuador", "Paraguay", "Uruguay"],
    "OFC":      ["New Zealand"],
}

# Normalização: nome canônico → nome na API-Football
NAME_MAP = {
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Turkey":                 "Türkiye",
    "Cape Verde":             "Cape Verde Islands",
    "DR Congo":               "Congo DR",
    "Curacao":                "Curaçao",
}

# Mapeamento inverso para padronizar tudo no nome canônico
INV_NAME_MAP = {v: k for k, v in NAME_MAP.items()}

STAT_COLS = [
    "shots", "shots_on_goal", "blocked_shots", "ball_possession",
    "fouls", "yellow_cards", "red_cards", "corners", "offsides",
    "passes_accurate", "saves",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_normalize() -> pd.DataFrame:
    """Carrega WCQ + hosts + AFCON, normaliza nomes, retorna DataFrame consolidado."""
    dfs = []

    wcq = pd.read_csv("data/raw/api_football_wcq_raw.csv")
    wcq["source"] = "WCQ"
    dfs.append(wcq)

    hosts_path = Path("data/raw/hosts_competitive_raw.csv")
    if hosts_path.exists():
        hosts = pd.read_csv(hosts_path)
        hosts["source"] = "HOSTS_COMP"
        dfs.append(hosts)

    afcon_path = Path("data/raw/caf_afcon_raw.csv")
    if afcon_path.exists():
        afcon = pd.read_csv(afcon_path)
        afcon["source"] = "AFCON"
        dfs.append(afcon)

    df = pd.concat(dfs, ignore_index=True)

    # Normaliza nomes para o canônico
    df["team"] = df["team"].map(lambda x: INV_NAME_MAP.get(x, x))
    df["opponent"] = df["opponent"].map(lambda x: INV_NAME_MAP.get(x, x))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def completeness(series: pd.Series) -> float:
    """Retorna % de valores não-nulos."""
    if len(series) == 0:
        return 0.0
    return round(series.notna().mean() * 100, 1)


def fmt_pct(v: float) -> str:
    return f"{v:5.1f}%"


# ---------------------------------------------------------------------------
# Análise por time
# ---------------------------------------------------------------------------

def analyze_team(team: str, df: pd.DataFrame) -> dict:
    sub = df[df["team"] == team].copy()
    n = len(sub)
    if n == 0:
        return {"team": team, "n_matches": 0}

    res = {
        "team": team,
        "n_matches": n,
        "date_min": sub["date"].min(),
        "date_max": sub["date"].max(),
        "date_range_days": (sub["date"].max() - sub["date"].min()).days if n > 1 else 0,
        "sources": sub["source"].value_counts().to_dict() if "source" in sub.columns else {},
        "competitions": sub["competition"].value_counts().to_dict(),
        "match_types": sub["match_type"].value_counts().to_dict() if "match_type" in sub.columns else {},
        "home_away": sub["home_away"].value_counts().to_dict() if "home_away" in sub.columns else {},
        "has_shot_data_pct": completeness(sub["has_shot_data"].replace(False, np.nan).replace(0, np.nan)) if "has_shot_data" in sub.columns else 0.0,
        "goals_completeness": completeness(sub["gols_marcados"]),
        "stats_completeness": {col: completeness(sub[col]) for col in STAT_COLS if col in sub.columns},
        "n_unique_opponents": sub["opponent"].nunique(),
        "goals_scored_mean": round(sub["gols_marcados"].mean(), 2) if sub["gols_marcados"].notna().any() else None,
        "goals_conceded_mean": round(sub["gols_sofridos"].mean(), 2) if sub["gols_sofridos"].notna().any() else None,
    }

    # Score de completude geral das stats
    pcts = list(res["stats_completeness"].values())
    res["avg_stats_completeness"] = round(np.mean(pcts), 1) if pcts else 0.0

    return res


# ---------------------------------------------------------------------------
# Categorias de qualidade
# ---------------------------------------------------------------------------

def quality_grade(r: dict) -> tuple[str, str]:
    """Retorna (grade, reason)."""
    n = r.get("n_matches", 0)
    avg = r.get("avg_stats_completeness", 0)

    if n == 0:
        return "F", "SEM DADOS"
    if n < 4:
        return "D", f"poucos jogos ({n})"
    if avg < 30:
        return "D", f"stats muito incompletas ({avg}%)"
    if n < 8 or avg < 60:
        return "C", f"cobertura parcial ({n} jogos, {avg}% stats)"
    if n < 12 or avg < 80:
        return "B", f"cobertura boa ({n} jogos, {avg}% stats)"
    return "A", f"cobertura excelente ({n} jogos, {avg}% stats)"


# ---------------------------------------------------------------------------
# Geração do relatório
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, results: list[dict]) -> str:
    lines = []
    sep = "=" * 78

    def add(s=""):
        lines.append(s)

    add(sep)
    add("  RELATÓRIO DE QUALIDADE DE DADOS — COPA DO MUNDO 2026 (48 SELEÇÕES)")
    add(sep)
    add(f"\n  Data de geração : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"  Total de linhas : {len(df):,} (todas as fontes consolidadas)")
    add(f"  Total de seleções cobertas: {sum(1 for r in results if r['n_matches'] > 0)}/48")

    # ---- Distribuição por qualidade
    grades = {r["team"]: quality_grade(r) for r in results}
    grade_counts = {}
    for g, _ in grades.values():
        grade_counts[g] = grade_counts.get(g, 0) + 1

    add("\n" + sep)
    add("  1. RESUMO EXECUTIVO")
    add(sep)
    for g in ["A", "B", "C", "D", "F"]:
        label = {"A": "Excelente", "B": "Boa", "C": "Parcial", "D": "Insuficiente", "F": "Sem dados"}[g]
        add(f"  Qualidade {g} ({label:12s}): {grade_counts.get(g,0):2d} seleções")

    # ---- Cobertura por confederação
    add("\n" + sep)
    add("  2. COBERTURA POR CONFEDERAÇÃO")
    add(sep)
    res_by_team = {r["team"]: r for r in results}

    for conf, teams in QUALIFIED_48.items():
        add(f"\n  [{conf}]")
        add(f"  {'Seleção':35s} {'Jogos':>6} {'Média Stats':>12} {'Gols':>6} {'Período':>22} {'Nota'}")
        add("  " + "-" * 72)
        for team in sorted(teams):
            r = res_by_team.get(team, {})
            n = r.get("n_matches", 0)
            avg = r.get("avg_stats_completeness", 0)
            gm = r.get("goals_scored_mean")
            d_min = r["date_min"].strftime("%Y-%m-%d") if r.get("date_min") and pd.notna(r["date_min"]) else "-"
            d_max = r["date_max"].strftime("%Y-%m-%d") if r.get("date_max") and pd.notna(r["date_max"]) else "-"
            periodo = f"{d_min} a {d_max}" if d_min != "-" else "sem dados"
            grade, _ = quality_grade(r)
            gols_str = f"{gm:.2f}" if gm is not None else " N/A"
            add(f"  {team:35s} {n:6d} {avg:11.1f}% {gols_str:>6}  {periodo:>22}  [{grade}]")

    # ---- Detalhe das estatísticas por time
    add("\n" + sep)
    add("  3. COMPLETUDE DAS ESTATÍSTICAS POR SELEÇÃO")
    add(sep)
    add(f"\n  {'Seleção':30s} {'Conf':8s} {'N':>5} " + " ".join(f"{c[:4]:>5}" for c in STAT_COLS))
    add("  " + "-" * 130)

    conf_map = {}
    for conf, teams in QUALIFIED_48.items():
        for t in teams:
            conf_map[t] = conf

    for r in sorted(results, key=lambda x: x.get("avg_stats_completeness", 0)):
        team = r["team"]
        conf = conf_map.get(team, "?")
        n = r.get("n_matches", 0)
        stats = r.get("stats_completeness", {})
        row_str = f"  {team:30s} {conf:8s} {n:5d} "
        for col in STAT_COLS:
            pct = stats.get(col, 0)
            marker = f"{pct:.0f}%" if pct > 0 else "  —  "
            row_str += f" {marker:>5}"
        add(row_str)

    # ---- Times problemáticos
    add("\n" + sep)
    add("  4. ALERTAS E PROBLEMAS IDENTIFICADOS")
    add(sep)

    alerts = []
    for r in results:
        team = r["team"]
        conf = conf_map.get(team, "?")
        n = r.get("n_matches", 0)
        avg = r.get("avg_stats_completeness", 0)
        grade, reason = quality_grade(r)

        if grade in ("D", "F"):
            alerts.append(("CRITICO", conf, team, n, avg, reason))
        elif grade == "C":
            alerts.append(("AVISO  ", conf, team, n, avg, reason))

        # Stats específicas faltando
        stats = r.get("stats_completeness", {})
        for col, pct in stats.items():
            if pct < 50 and n >= 5:
                alerts.append(("STAT   ", conf, team, n, pct, f"'{col}' apenas {pct}% completo"))

    alerts.sort(key=lambda x: (0 if x[0] == "CRITICO" else 1 if x[0] == "AVISO" else 2, x[1], x[2]))

    if alerts:
        add(f"\n  {'Nível':8s} {'Conf':8s} {'Seleção':30s} {'N':>5} {'%':>6}  Motivo")
        add("  " + "-" * 80)
        for level, conf, team, n, pct, reason in alerts:
            add(f"  {level} {conf:8s} {team:30s} {n:5d} {pct:5.1f}%  {reason}")
    else:
        add("\n  Nenhum alerta critico identificado.")

    # ---- Estatísticas globais
    add("\n" + sep)
    add("  5. ESTATÍSTICAS GLOBAIS DO DATASET")
    add(sep)

    qualified_teams = [t for conf_teams in QUALIFIED_48.values() for t in conf_teams]
    df_48 = df[df["team"].isin(qualified_teams)]

    add(f"\n  Total de registros (48 seleções)  : {len(df_48):,}")
    add(f"  Total de partidas únicas          : {df_48['match_id'].nunique():,}")
    add(f"  Período coberto                   : {df_48['date'].min().strftime('%Y-%m-%d')} a {df_48['date'].max().strftime('%Y-%m-%d')}")
    add(f"  Média de jogos por seleção        : {len(df_48)/48:.1f}")

    add(f"\n  Completude das colunas-chave (todas as 48 seleções):")
    for col in ["gols_marcados", "gols_sofridos"] + STAT_COLS:
        if col in df_48.columns:
            pct = completeness(df_48[col])
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            add(f"    {col:22s} [{bar}] {pct:5.1f}%")

    # Distribuição por tipo de competição
    add(f"\n  Distribuição por tipo de competição:")
    if "competition" in df_48.columns:
        for comp, cnt in df_48["competition"].value_counts().items():
            add(f"    {comp:40s}: {cnt:5d} registros")

    # ---- Recomendações
    add("\n" + sep)
    add("  6. RECOMENDAÇÕES")
    add(sep)
    add("""
  [R1] USA / México / Canadá: dados de WCQ ausentes (países-sede isentos).
       → Compensados por Copa América 2024, CONCACAF Nations League e Gold Cup.
       → Considerar peso maior (weight=1.0) para CONCACAF NL pois é competição
         de alto nível entre os mesmos times da confederação.

  [R2] Seleções com stats incompletas (ball_possession, passes_accurate, saves):
       → Estas colunas ficam nulas para OFC e algumas partidas antigas.
       → Estratégia: imputar mediana por confederação antes da modelagem.

  [R3] New Zealand (OFC): apenas 5 jogos WCQ com stats limitadas.
       → Suplementar com dados do OFC Nations Cup ou amistosos via API-Football.

  [R4] Partidas de Gold Cup: considerar peso menor (weight=0.8) pois inclui
       times não-qualificados na fase de grupos.

  [R5] Verificar duplicatas de match_id entre fontes antes de modelar.
    """)

    add(sep)
    add("  FIM DO RELATÓRIO")
    add(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Carregando dados...")
    df = load_and_normalize()

    print("Analisando 48 seleções...")
    qualified_teams = [t for conf_teams in QUALIFIED_48.values() for t in conf_teams]
    results = [analyze_team(team, df) for team in qualified_teams]

    report = generate_report(df, results)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nRelatório salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
