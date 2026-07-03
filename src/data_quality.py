"""
Diagnóstico de qualidade dos dados — Copa 2026
Analisa data/raw/all_confederations_raw.csv

Se all_confederations_raw.csv não existir, tenta montar o dataset dos CSVs
individuais por confederação + conmebol_match_stats_raw.csv + friendlies_raw.csv.

Saída: outputs/data_quality_report.txt  +  resumo executivo no terminal.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from datetime import datetime

import difflib
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAW_DIR    = Path("data/raw")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "data_quality_report.txt"

ALL_CSV      = RAW_DIR / "all_confederations_raw.csv"
CONMEBOL_CSV = RAW_DIR / "conmebol_match_stats_raw.csv"
FRIENDLIES_CSV = RAW_DIR / "friendlies_raw.csv"

# Confederações usadas em fetch_all_confederations.py
CONFEDERATION_KEYS = ["UEFA", "CONCACAF", "AFC", "CAF", "CONMEBOL", "OFC"]

# ---------------------------------------------------------------------------
# Colunas de estatísticas
# ---------------------------------------------------------------------------
STAT_COLS_SHORT = ["sh", "sot", "gls", "ast", "tklw", "int", "fls", "crdy", "crdr"]
# O dataset usa prefixo "performance_"
STAT_COLS_PERF  = [f"performance_{s}" for s in STAT_COLS_SHORT]

# Limiares de outlier ao nível de TIME-JOGO (soma dos jogadores)
OUTLIER_THRESHOLDS_TEAM = {
    "performance_sh":   40,   # 40 chutes por time num jogo seria absurdo
    "performance_sot":  20,
    "performance_gls":   8,   # 8+ gols por time num jogo
    "performance_ast":   8,
    "performance_tklw": 25,
    "performance_int":  20,
    "performance_fls":  25,
    "performance_crdy":  6,
    "performance_crdr":  3,
}

# ---------------------------------------------------------------------------
# 48 times classificados para a Copa 2026 (atualize se necessário)
# Nomes no formato FBref — verificar com team_name no dataset
# ---------------------------------------------------------------------------
COPA2026_TEAMS: list[str] = [
    # CONMEBOL (6)
    "Argentina", "Brazil", "Colombia", "Uruguay", "Ecuador", "Venezuela",
    # UEFA (16) — nomes FBref
    "Spain", "Germany", "France", "England", "Portugal", "Netherlands",
    "Italy", "Türkiye", "Austria", "Scotland", "Switzerland", "Croatia",
    "Serbia", "Hungary", "Slovenia", "Romania",
    # AFC (8) — FBref: "Korea Republic", "IR Iran"
    "Japan", "Korea Republic", "IR Iran", "Iraq", "Jordan", "Australia",
    "Uzbekistan", "Saudi Arabia",
    # CONCACAF (6) — USA/Canada/México são co-anfitriões
    "United States", "Mexico", "Canada", "Panama", "Honduras", "Jamaica",
    # CAF (9) — FBref: "Côte d'Ivoire", "Congo DR"
    "Morocco", "Senegal", "Egypt", "Nigeria", "Cameroon", "Côte d'Ivoire",
    "South Africa", "Tunisia", "Congo DR",
    # OFC (1)
    "New Zealand",
    # Playoffs inter-confederações (2) — ajustar quando confirmados
    # "Indonesia", "Bahrain",
]

# ---------------------------------------------------------------------------
# Helpers de escrita
# ---------------------------------------------------------------------------
_report_lines: list[str] = []


def _w(*args):
    """Escreve linha no buffer do relatório."""
    line = " ".join(str(a) for a in args)
    _report_lines.append(line)


def _section(title: str):
    _w()
    _w("=" * 70)
    _w(f"  {title}")
    _w("=" * 70)


def _sub(title: str):
    _w()
    _w(f"  -- {title} --")


# ---------------------------------------------------------------------------
# Carregamento do dataset
# ---------------------------------------------------------------------------

def _resolve_stat_cols(df: pd.DataFrame) -> list[str]:
    """Retorna as colunas de stat que existem no df (prefixo performance_ ou nu)."""
    perf = [c for c in STAT_COLS_PERF if c in df.columns]
    if perf:
        return perf
    bare = [c for c in STAT_COLS_SHORT if c in df.columns]
    return bare


def _add_missing_meta(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Garante colunas de metadados mínimas."""
    if "confederation" not in df.columns:
        # Tenta inferir da coluna competition
        if "competition" in df.columns:
            mapping = {
                "CONMEBOL": "CONMEBOL", "Copa América": "CONMEBOL",
                "UEFA": "UEFA",         "Euro": "UEFA",   "Nations League": "UEFA",
                "CONCACAF": "CONCACAF", "Gold Cup": "CONCACAF",
                "AFC": "AFC",           "Asian Cup": "AFC",
                "CAF": "CAF",           "AFCON": "CAF",   "Africa Cup": "CAF",
                "OFC": "OFC",
                "friendly": "FRIENDLY",
            }
            def _infer_conf(comp):
                comp = str(comp)
                for k, v in mapping.items():
                    if k.lower() in comp.lower():
                        return v
                return "UNKNOWN"
            df["confederation"] = df["competition"].apply(_infer_conf)
        else:
            df["confederation"] = source_label

    if "competition" not in df.columns:
        df["competition"] = source_label
    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    return df


def load_dataset() -> pd.DataFrame:
    """Carrega o dataset principal; fallback para CSVs individuais."""
    dfs: list[pd.DataFrame] = []

    if ALL_CSV.exists():
        print(f"  Carregando {ALL_CSV} ...")
        df = pd.read_csv(ALL_CSV, low_memory=False)
        print(f"  → {len(df):,} linhas, {df['match_id'].nunique() if 'match_id' in df.columns else '?'} match_ids únicos")
        return df

    print(f"  AVISO: {ALL_CSV} não encontrado — montando a partir de CSVs disponíveis")

    # Confederation CSVs (UEFA_raw.csv, etc.)
    for conf in CONFEDERATION_KEYS:
        p = RAW_DIR / f"{conf}_raw.csv"
        if p.exists() and p.stat().st_size > 0:
            df = pd.read_csv(p, low_memory=False)
            df = _add_missing_meta(df, conf)
            dfs.append(df)
            print(f"    {conf}_raw.csv: {len(df):,} linhas")

    # CONMEBOL WCQ legacy
    if CONMEBOL_CSV.exists():
        df = pd.read_csv(CONMEBOL_CSV, low_memory=False)
        df = _add_missing_meta(df, "CONMEBOL")
        if "competition" not in df.columns or df["competition"].isna().all():
            df["competition"] = "CONMEBOL WCQ 2026"
        dfs.append(df)
        print(f"    conmebol_match_stats_raw.csv: {len(df):,} linhas")

    # Friendlies
    if FRIENDLIES_CSV.exists():
        df = pd.read_csv(FRIENDLIES_CSV, low_memory=False)
        df = _add_missing_meta(df, "FRIENDLY")
        dfs.append(df)
        print(f"    friendlies_raw.csv: {len(df):,} linhas")

    if not dfs:
        print("  ERRO: Nenhum CSV de dados encontrado em data/raw/. Execute fetch_all_confederations.py primeiro.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  → Total: {len(combined):,} linhas, {combined['match_id'].nunique() if 'match_id' in combined.columns else '?'} match_ids")
    return combined


# ---------------------------------------------------------------------------
# BLOCO 1 — Cobertura por confederação
# ---------------------------------------------------------------------------

def bloco1_cobertura(df: pd.DataFrame):
    _section("BLOCO 1 — COBERTURA POR CONFEDERAÇÃO")

    if "match_id" not in df.columns:
        _w("  ERRO: coluna match_id não encontrada.")
        return

    # Jogos únicos por confederação + competição
    _sub("Jogos únicos por confederação e competição")
    keys = ["confederation", "competition"] if "competition" in df.columns else ["confederation"]
    sched = df.drop_duplicates(subset=["match_id"])[["match_id"] + keys + (["year"] if "year" in df.columns else [])]
    by_comp = (
        sched.groupby(keys, dropna=False)["match_id"]
        .nunique()
        .reset_index(name="jogos_unicos")
        .sort_values(["confederation", "jogos_unicos"], ascending=[True, False])
    )
    _w(by_comp.to_string(index=False))
    _w(f"\n  TOTAL jogos únicos: {df['match_id'].nunique()}")

    # Times únicos por confederação
    _sub("Times únicos por confederação")
    if "team_name" in df.columns and "confederation" in df.columns:
        team_counts = (
            df.groupby("confederation", dropna=False)["team_name"]
            .apply(lambda s: s.dropna().astype(str).nunique())
            .reset_index(name="times_unicos")
            .sort_values("times_unicos", ascending=False)
        )
        _w(team_counts.to_string(index=False))

    # Distribuição por ano
    _sub("Distribuição de jogos por ano")
    if "year" in sched.columns:
        by_year = (
            sched.groupby(["confederation", "year"], dropna=False)["match_id"]
            .nunique()
            .unstack(fill_value=0)
        )
        _w(by_year.to_string())
    elif "date" in df.columns:
        sched2 = sched.copy()
        sched2["year"] = pd.to_datetime(df.drop_duplicates("match_id")["date"], errors="coerce").dt.year
        by_year = sched2.groupby(["confederation", "year"], dropna=False)["match_id"].nunique().unstack(fill_value=0)
        _w(by_year.to_string())


# ---------------------------------------------------------------------------
# BLOCO 2 — Qualidade das features
# ---------------------------------------------------------------------------

def bloco2_features(df: pd.DataFrame) -> list[str]:
    """Retorna lista de alertas críticos."""
    _section("BLOCO 2 — QUALIDADE DAS FEATURES")
    alerts: list[str] = []

    stat_cols = _resolve_stat_cols(df)
    if not stat_cols:
        _w("  ERRO: Nenhuma coluna de stat encontrada.")
        return alerts

    _w(f"  Colunas de stat encontradas: {stat_cols}")

    # 2a. Percentual de nulos por confederação
    _sub("Percentual de nulos (%) por confederação")
    if "confederation" in df.columns:
        null_pct = (
            df.groupby("confederation", dropna=False)[stat_cols]
            .apply(lambda g: g.isna().mean() * 100)
            .round(1)
        )
        _w(null_pct.to_string())

        # Classificação: completo (<5%), parcial (5-50%), esparso (>50%)
        _sub("Classificação por completude (coluna 'gls')")
        gls_col = next((c for c in stat_cols if c.endswith("gls")), None)
        sh_col  = next((c for c in stat_cols if c.endswith("_sh") or c == "sh"), None)

        if gls_col and sh_col and "confederation" in df.columns:
            for conf, g in df.groupby("confederation", dropna=False):
                sh_null  = g[sh_col].isna().mean() * 100
                gls_null = g[gls_col].isna().mean() * 100
                if sh_null < 5:
                    status = "COMPLETO"
                elif sh_null < 50:
                    status = "PARCIAL"
                else:
                    status = "ESPARSO"
                    alerts.append(f"[BLOCO2] {conf}: sh {sh_null:.0f}% nulo — ESPARSO (pouco útil para features)")
                _w(f"    {conf:<14} sh_null={sh_null:5.1f}%  gls_null={gls_null:5.1f}%  → {status}")
    else:
        null_pct = df[stat_cols].isna().mean() * 100
        _w(null_pct.round(1).to_string())

    # 2b. Distribuição a nível de TIME-JOGO
    _sub("Distribuição de stats ao nível TEAM-MATCH (soma por time por jogo)")
    id_cols = [c for c in ["match_id", "team_name", "team_side", "confederation"] if c in df.columns]
    if "match_id" in df.columns and "team_name" in df.columns:
        agg_stat_cols = [c for c in stat_cols if c in df.columns]
        team_match = (
            df.groupby([c for c in id_cols if c != "confederation"], dropna=False)[agg_stat_cols]
            .sum(min_count=1)
            .reset_index()
        )
        # Junta confederation de volta
        if "confederation" in df.columns:
            conf_map = df.drop_duplicates("match_id").set_index("match_id")["confederation"].to_dict()
            team_match["confederation"] = team_match["match_id"].map(conf_map)

        desc_rows = []
        for col in agg_stat_cols:
            s = team_match[col].dropna()
            if len(s) == 0:
                continue
            desc_rows.append({
                "stat": col.replace("performance_", ""),
                "n":    len(s),
                "mean": round(s.mean(), 2),
                "std":  round(s.std(), 2),
                "min":  s.min(),
                "p25":  s.quantile(0.25),
                "p50":  s.median(),
                "p75":  s.quantile(0.75),
                "max":  s.max(),
            })
        if desc_rows:
            _w(pd.DataFrame(desc_rows).to_string(index=False))

        # 2c. Outliers absurdos
        _sub("Outliers absurdos (valores acima do limiar por time-jogo)")
        found_outlier = False
        for col, threshold in OUTLIER_THRESHOLDS_TEAM.items():
            if col not in team_match.columns:
                continue
            outliers = team_match[team_match[col] > threshold][
                [c for c in ["match_id", "team_name", "confederation", col] if c in team_match.columns]
            ]
            if len(outliers) > 0:
                found_outlier = True
                _w(f"\n  {col} > {threshold}:")
                _w(outliers.to_string(index=False))
                alerts.append(f"[BLOCO2] {len(outliers)} registros com {col} > {threshold} (possível erro de coleta)")
        if not found_outlier:
            _w("  Nenhum outlier absurdo detectado.")

    return alerts


# ---------------------------------------------------------------------------
# BLOCO 3 — Consistência de nomes
# ---------------------------------------------------------------------------

def bloco3_nomes(df: pd.DataFrame) -> list[str]:
    _section("BLOCO 3 — CONSISTÊNCIA DE NOMES")
    alerts: list[str] = []

    # Coleta todos os nomes únicos de times
    team_cols = [c for c in ["home_team", "away_team", "team_name"] if c in df.columns]
    all_teams: set[str] = set()
    for col in team_cols:
        all_teams.update(
            t for t in df[col].dropna().astype(str).unique()
            if t not in ("", "nan", "?")
        )
    all_teams_list = sorted(all_teams)

    _sub(f"Todos os nomes de times únicos ({len(all_teams_list)} total)")
    _w("\n".join(f"    {t}" for t in all_teams_list))

    # Possíveis duplicatas por similaridade
    _sub("Possíveis duplicatas (similaridade ≥ 0.82, nomes diferentes)")
    pairs_found = []
    for i, t1 in enumerate(all_teams_list):
        for t2 in all_teams_list[i + 1:]:
            ratio = difflib.SequenceMatcher(None, t1.lower(), t2.lower()).ratio()
            if ratio >= 0.82:
                pairs_found.append((ratio, t1, t2))

    if pairs_found:
        pairs_found.sort(reverse=True)
        for ratio, t1, t2 in pairs_found:
            _w(f"    {ratio:.2f}  |  '{t1}'  vs  '{t2}'")
            alerts.append(f"[BLOCO3] Possível duplicata: '{t1}' vs '{t2}' (sim={ratio:.2f})")
    else:
        _w("  Nenhuma duplicata suspeita encontrada.")

    # Cruzar com lista dos 48 classificados
    _sub("Times da Copa 2026 NÃO encontrados no dataset")
    not_found = []
    found_as = {}
    for copa_team in COPA2026_TEAMS:
        # Busca exata
        if copa_team in all_teams:
            found_as[copa_team] = copa_team
            continue
        # Busca por substring case-insensitive
        matches = [t for t in all_teams_list if copa_team.lower() in t.lower() or t.lower() in copa_team.lower()]
        if matches:
            found_as[copa_team] = matches[0]
            continue
        # Busca fuzzy
        close = difflib.get_close_matches(copa_team, all_teams_list, n=1, cutoff=0.72)
        if close:
            found_as[copa_team] = f"≈ {close[0]}"
            continue
        not_found.append(copa_team)

    _w(f"\n  Times encontrados no dataset: {len(found_as)}/{len(COPA2026_TEAMS)}")
    if not_found:
        _w(f"  NÃO ENCONTRADOS ({len(not_found)}):")
        for t in not_found:
            _w(f"    - {t}")
            alerts.append(f"[BLOCO3] Time Copa 2026 ausente no dataset: '{t}'")
    else:
        _w("  Todos os times classificados encontrados!")

    # Mapeamentos suspeitos (encontrado via fuzzy/substring)
    fuzzy_maps = [(k, v) for k, v in found_as.items() if k != v]
    if fuzzy_maps:
        _sub("Mapeamentos por similaridade (verificar se o nome está correto no FBref)")
        for copa, dataset_name in sorted(fuzzy_maps):
            _w(f"    Copa: '{copa}'  →  Dataset: '{dataset_name}'")

    return alerts


# ---------------------------------------------------------------------------
# BLOCO 4 — Volume por time
# ---------------------------------------------------------------------------

def bloco4_volume(df: pd.DataFrame) -> list[str]:
    _section("BLOCO 4 — VOLUME POR TIME")
    alerts: list[str] = []

    if "team_name" not in df.columns or "match_id" not in df.columns:
        _w("  ERRO: colunas team_name ou match_id não encontradas.")
        return alerts

    stat_cols = _resolve_stat_cols(df)
    sh_col  = next((c for c in stat_cols if c.endswith("_sh") or c == "sh"), None)
    gls_col = next((c for c in stat_cols if c.endswith("gls")), None)

    # Agrega por time-jogo
    def _has_sh(g):
        if sh_col and sh_col in g.columns:
            return g[sh_col].notna().any()
        return False

    def _has_gls(g):
        if gls_col and gls_col in g.columns:
            return g[gls_col].notna().any()
        return False

    team_match_df = df.groupby(["team_name", "match_id"], dropna=False)

    records = []
    for (team, mid), g in team_match_df:
        records.append({
            "team":     team,
            "match_id": mid,
            "sh_ok":    _has_sh(g),
            "gls_ok":   _has_gls(g),
        })
    tm = pd.DataFrame(records)

    summary = tm.groupby("team", dropna=False).agg(
        jogos_total   = ("match_id", "nunique"),
        jogos_sh_ok   = ("sh_ok",   "sum"),
        jogos_gls_only= ("gls_ok",  lambda s: (s & ~tm.loc[s.index, "sh_ok"]).sum()),
    ).reset_index()
    summary = summary.sort_values("jogos_sh_ok", ascending=False)

    _sub("Todos os times — jogos com sh completo vs só gls/ast")
    _w(summary.to_string(index=False))

    # Times problemáticos: < 5 jogos com sh
    MIN_SH_GAMES = 5
    problematic = summary[summary["jogos_sh_ok"] < MIN_SH_GAMES].sort_values("jogos_sh_ok")
    _sub(f"Times com menos de {MIN_SH_GAMES} jogos com sh completo (problemáticos para janela móvel)")
    if len(problematic) > 0:
        _w(problematic.to_string(index=False))
        for _, row in problematic.iterrows():
            alerts.append(
                f"[BLOCO4] '{row['team']}': apenas {int(row['jogos_sh_ok'])} jogos com sh "
                f"(total={int(row['jogos_total'])})"
            )
    else:
        _w(f"  Todos os times têm ≥ {MIN_SH_GAMES} jogos com sh completo.")

    # Top 10 / Bottom 10
    _sub("Top 10 times com mais dados (jogos_sh_ok)")
    _w(summary.head(10).to_string(index=False))

    _sub("Bottom 10 times com menos dados")
    _w(summary.tail(10).to_string(index=False))

    return alerts


# ---------------------------------------------------------------------------
# BLOCO 5 — Duplicatas e integridade
# ---------------------------------------------------------------------------

def bloco5_integridade(df: pd.DataFrame) -> list[str]:
    _section("BLOCO 5 — DUPLICATAS E INTEGRIDADE")
    alerts: list[str] = []

    if "match_id" not in df.columns:
        _w("  ERRO: coluna match_id não encontrada.")
        return alerts

    # 5a. match_ids duplicados (mesmo match_id com dados conflitantes)
    _sub("match_ids duplicados no dataset (rows com o mesmo match_id por team_name/stat_type/player)")
    key_cols = [c for c in ["match_id", "team_name", "stat_type", "player"] if c in df.columns]
    dups = df[df.duplicated(subset=key_cols, keep=False)]
    if len(dups) > 0:
        n_matches = dups["match_id"].nunique()
        _w(f"  {len(dups)} linhas duplicadas em {n_matches} match_ids")
        _w(dups[key_cols + (["date", "score"] if "score" in df.columns else [])].head(20).to_string(index=False))
        alerts.append(f"[BLOCO5] {len(dups)} linhas duplicadas em {n_matches} match_ids")
    else:
        _w("  Nenhum duplicado exato encontrado.")

    # 5b. Mesmo par home/away/date mais de uma vez (match_ids diferentes)
    _sub("Par home_team/away_team/date com match_ids diferentes (jogo duplicado)")
    trip_cols = [c for c in ["home_team", "away_team", "date"] if c in df.columns]
    if len(trip_cols) == 3:
        sched = df.drop_duplicates("match_id")[trip_cols + ["match_id"]]
        dup_trips = sched[sched.duplicated(subset=trip_cols, keep=False)]
        if len(dup_trips) > 0:
            _w(f"  {len(dup_trips)} registros suspeitos:")
            _w(dup_trips.sort_values(trip_cols).to_string(index=False))
            alerts.append(f"[BLOCO5] {dup_trips['match_id'].nunique()} match_ids com mesmo home/away/date")
        else:
            _w("  Nenhum par home/away/date duplicado.")
    else:
        _w(f"  Colunas trip insuficientes: {trip_cols}")

    # 5c. Consistência score vs gls
    _sub("Consistência: soma de gls dos jogadores vs score do jogo")
    gls_col = next(
        (c for c in _resolve_stat_cols(df) if c.endswith("gls")),
        None
    )
    score_col = "score" if "score" in df.columns else None

    if gls_col and score_col and "team_side" in df.columns:
        def _parse_score(s: str) -> tuple[int | None, int | None]:
            """Parseia score tipo '2–1', '2-1', '2:1'."""
            m = re.search(r"(\d+)\s*[–\-:]\s*(\d+)", str(s))
            if m:
                return int(m.group(1)), int(m.group(2))
            return None, None

        tm_gls = (
            df.groupby(["match_id", "team_side", "score"], dropna=False)[gls_col]
            .sum(min_count=1)
            .reset_index(name="gls_sum")
        )
        mismatch_rows = []
        for _, row in tm_gls.iterrows():
            home_g, away_g = _parse_score(row["score"])
            if home_g is None:
                continue
            expected = home_g if row["team_side"] == "home" else away_g
            actual   = row["gls_sum"]
            if pd.isna(actual):
                continue
            if abs(int(actual) - expected) > 0:
                mismatch_rows.append({
                    "match_id":  row["match_id"],
                    "team_side": row["team_side"],
                    "score":     row["score"],
                    "expected_gls": expected,
                    "actual_gls":   int(actual),
                    "diff":      int(actual) - expected,
                })
        if mismatch_rows:
            mm = pd.DataFrame(mismatch_rows)
            _w(f"  {len(mm)} inconsistências gls vs score (diff != 0):")
            _w(mm.to_string(index=False))
            alerts.append(
                f"[BLOCO5] {len(mm)} inconsistências gls vs score "
                f"(pode indicar gols contra, subs NaN, ou erros de parsing)"
            )
        else:
            _w("  Soma de gls bate com score em todos os jogos verificados.")
    else:
        _w(f"  Pulado: gls_col={gls_col}, score_col={score_col}, team_side={'team_side' in df.columns}")

    return alerts


# ---------------------------------------------------------------------------
# Resumo executivo
# ---------------------------------------------------------------------------

def print_executive_summary(df: pd.DataFrame, all_alerts: list[str]):
    total_rows   = len(df)
    total_matches = df["match_id"].nunique() if "match_id" in df.columns else "?"
    total_teams = set()
    for col in ["home_team", "away_team", "team_name"]:
        if col in df.columns:
            total_teams.update(df[col].dropna().astype(str).unique())
    total_teams.discard("nan"); total_teams.discard("?"); total_teams.discard("")

    print()
    print("=" * 70)
    print("  RESUMO EXECUTIVO — DATA QUALITY")
    print("=" * 70)
    print(f"  Dataset:         {total_rows:,} linhas")
    print(f"  Jogos únicos:    {total_matches}")
    print(f"  Times únicos:    {len(total_teams)}")
    if "confederation" in df.columns:
        print(f"  Confederações:   {sorted(df['confederation'].dropna().unique())}")
    if "year" in df.columns:
        years = sorted(df["year"].dropna().unique().astype(int))
        print(f"  Anos cobertos:   {years}")
    print()
    if all_alerts:
        print(f"  ALERTAS ({len(all_alerts)}) — resolver antes de modelar:")
        for a in all_alerts:
            print(f"    ⚠  {a}")
    else:
        print("  Nenhum alerta crítico. Dados prontos para feature engineering.")
    print()
    print("  Sugestões de correção:")
    bloco3_alerts = [a for a in all_alerts if "BLOCO3" in a]
    bloco4_alerts = [a for a in all_alerts if "BLOCO4" in a]
    bloco5_alerts = [a for a in all_alerts if "BLOCO5" in a]
    bloco2_alerts = [a for a in all_alerts if "BLOCO2" in a]
    if bloco3_alerts:
        print("    [Nome] Padronizar nomes de times antes de build_features.py — criar")
        print("           um dicionário de renomeação (ex: 'Korea Republic' → 'South Korea').")
    if bloco4_alerts:
        print("    [Volume] Para times com < 5 jogos com sh: considerar imputação pela")
        print("           média da confederação ou excluir da janela móvel de features.")
    if bloco5_alerts:
        print("    [Integridade] Investigar match_ids duplicados e inconsistências de score.")
        print("           Possíveis causas: pênaltis perdidos, gols contra, rows NaN tratadas como 0.")
    if bloco2_alerts:
        print("    [Nulos] Confederações com stats esparsos: criar flag 'has_shot_data'")
        print("           e imputar stats faltantes pela média da confederação.")
    if not all_alerts:
        print("    Nenhuma ação necessária — prosseguir com build_features.py.")
    print()
    print(f"  Relatório completo salvo em: {REPORT_PATH}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 70)
    print("  DATA QUALITY — Copa 2026")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df = load_dataset()

    # Normaliza datas
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Garante coluna year
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = df["date"].dt.year

    # Executa os 5 blocos
    bloco1_cobertura(df)
    alerts2 = bloco2_features(df)
    alerts3 = bloco3_nomes(df)
    alerts4 = bloco4_volume(df)
    alerts5 = bloco5_integridade(df)

    all_alerts = alerts2 + alerts3 + alerts4 + alerts5

    # Rodapé do relatório
    _section("METADADOS DO RELATÓRIO")
    _w(f"  Gerado em:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _w(f"  Arquivo fonte: {ALL_CSV if ALL_CSV.exists() else 'fallback de CSVs individuais'}")
    _w(f"  Total linhas:  {len(df):,}")
    _w(f"  Total alertas: {len(all_alerts)}")

    # Salva relatório
    REPORT_PATH.write_text(
        "\n".join(_report_lines),
        encoding="utf-8"
    )

    # Imprime resumo executivo no terminal
    print_executive_summary(df, all_alerts)


if __name__ == "__main__":
    main()
