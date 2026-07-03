"""
PASSO 1 — Normaliza nomes de times em api_football_wcq_raw.csv
PASSO 2 — Integra com friendlies_raw.csv → data/raw/full_dataset_raw.csv

Schema unificado (1 linha por time por jogo):
  match_id, date, competition, confederation, match_type, season, home_away,
  team, opponent, gols_marcados, gols_sofridos,
  shots, shots_on_goal, blocked_shots, ball_possession, fouls, yellow_cards,
  red_cards, corners, offsides, passes_accurate, saves, ast,
  has_shot_data, weight

WCQ      : weight=1.0, match_type=WCQ,      stats completos (tem shots/blocked/etc.)
Friendly : weight=1.0, match_type=friendly, só gols/ast/fouls/cards/offsides
           (distinção WCQ vs friendly é feita pelo dummy match_type_wcq no modelo)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WCQ_CSV        = Path("data/raw/api_football_wcq_raw.csv")
EXTRAS_CSV     = Path("data/raw/concacaf_extra_raw.csv")   # legado (pode não existir)
HOSTS_CSV      = Path("data/raw/hosts_competitive_raw.csv") # Copa América + CONCACAF NL + Gold Cup
AFCON_CSV      = Path("data/raw/caf_afcon_raw.csv")         # AFCON 2023 + 2025
JUN2026_CSV    = Path("data/raw/friendlies_jun2026_raw.csv")  # Amistosos pré-Copa jun/2026
FRIENDLIES_CSV = Path("data/raw/friendlies_raw.csv")
OUTPUT_CSV     = Path("data/raw/full_dataset_raw.csv")

# Mapeamento canônico de nomes de times (variantes → padrão)
# Aplicado a team e opponent em ambas as fontes
NAME_MAP: dict[str, str] = {
    # API-Football
    "USA":                    "United States",
    "México":                 "Mexico",
    "Curaçao":                "Curacao",
    "Côte d'Ivoire":          "Ivory Coast",
    "Korea Republic":         "South Korea",
    "Korea DPR":              "North Korea",
    "IR Iran":                "Iran",
    "FYR Macedonia":          "North Macedonia",
    "Bosnia & Herzegovina":   "Bosnia and Herzegovina",
    "Türkiye":                "Turkey",
    "Cape Verde Islands":     "Cape Verde",
    "Congo DR":               "DR Congo",
    # Qualidade — CORREÇÃO 1 (auditoria model_dataset v11)
    "Congo":                  "DR Congo",            # CAF WCQ: "Congo" → DR Congo
    "Bosnia-Herzegovina":     "Bosnia and Herzegovina",
    "Equ. Guinea":            "Equatorial Guinea",
    # FBref
}

NEUTRAL_WCQ_IDS: set[str] = {
    "1479253",  # Iraq home em Jeddah (2025-10-11)
    "1347242",  # South Africa home em Marrakech (2025-12-22)
    "1347241",  # Egypt home em Agadir (2025-12-22)
    "1347244",  # Tunisia home em Rabat (2025-12-23)
    "1347247",  # Algeria home em Rabat (2025-12-24)
    "1347249",  # Ivory Coast home em Marrakech (2025-12-24)
    "1347253",  # Egypt home em Agadir (2025-12-26)
    "1347259",  # Algeria home em Rabat (2025-12-28)
    "1347261",  # Ivory Coast home em Marrakech (2025-12-28)
    "1501837",  # South Africa home em Rabat (2026-01-04)
    "1501841",  # Ivory Coast home em Marrakech (2026-01-06)
    "1503402",  # Egypt home em Agadir (2026-01-10)
}

COPA2026_TEAMS = {
    # CONMEBOL (6)
    "Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay", "Paraguay",
    # UEFA (16)
    "Germany", "France", "Spain", "England", "Netherlands", "Portugal",
    "Belgium", "Austria", "Switzerland", "Croatia",
    "Norway", "Sweden", "Czech Republic", "Turkey",
    "Scotland", "Bosnia and Herzegovina",
    # AFC (9)
    "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia",
    "Uzbekistan", "Jordan", "Iraq", "Qatar",
    # CAF (10)
    "Morocco", "Senegal", "Egypt", "Ghana", "Cape Verde",
    "DR Congo", "Ivory Coast", "South Africa", "Algeria", "Tunisia",
    # CONCACAF (6, co-hosts incluídos)
    "United States", "Mexico", "Canada", "Panama", "Curacao", "Haiti",
    # OFC (1)
    "New Zealand",
}

OUTPUT_COLS = [
    "match_id", "date", "competition", "confederation", "match_type", "season",
    "home_away", "team", "opponent",
    "gols_marcados", "gols_sofridos",
    "shots", "shots_on_goal", "blocked_shots", "ball_possession",
    "fouls", "yellow_cards", "red_cards", "corners", "offsides", "passes_accurate", "saves",
    "ast",
    "has_shot_data", "weight",
    "home_advantage",
]


def _normalize(series: pd.Series) -> pd.Series:
    return series.map(lambda x: NAME_MAP.get(str(x), str(x)))


# ---------------------------------------------------------------------------
# PASSO 1a — WCQ (já em nível de time)
# ---------------------------------------------------------------------------

def load_tournament_extras() -> list[pd.DataFrame]:
    """
    Carrega fontes competitivas adicionais (mesmo schema que WCQ):
      - hosts_competitive_raw.csv  (USA/México/Canadá: Copa América, CONCACAF NL, Gold Cup)
      - caf_afcon_raw.csv          (10 times africanos: AFCON 2023 + 2025)
    Adiciona ast=NaN e weight=1.0 onde ausentes.
    """
    frames = []
    for path, label in [(HOSTS_CSV, "HOSTS"), (AFCON_CSV, "AFCON"),
                        (JUN2026_CSV, "JUN2026")]:
        if not path.exists():
            print(f"  [{label}] {path.name} nao encontrado — ignorando.")
            continue
        df = pd.read_csv(path)
        df["match_id"] = df["match_id"].astype(str)
        df["ast"]    = np.nan
        if "weight" not in df.columns:
            df["weight"] = 1.0
        df["team"]     = _normalize(df["team"])
        df["opponent"] = _normalize(df["opponent"])
        # red_cards: null → 0 onde has_shot_data=True (null significa "sem cartão")
        if "red_cards" in df.columns and "has_shot_data" in df.columns:
            mask = df["has_shot_data"].map(
                {"True": True, "False": False, True: True, False: False, 1: True, 0: False}
            ).fillna(False)
            df.loc[mask & df["red_cards"].isna(), "red_cards"] = 0.0
        df["home_advantage"] = 0  # torneios em campo neutro
        print(f"  [{label}] {path.name}: {df['match_id'].nunique()} jogos, {len(df)} linhas")
        frames.append(df[OUTPUT_COLS])
    return frames


def load_wcq() -> pd.DataFrame:
    df = pd.read_csv(WCQ_CSV)
    df["weight"]   = 1.0
    df["ast"]      = np.nan
    df["match_id"] = df["match_id"].astype(str)

    before = set(df["team"].tolist()) | set(df["opponent"].tolist())
    df["team"]     = _normalize(df["team"])
    df["opponent"] = _normalize(df["opponent"])
    after = set(df["team"].tolist()) | set(df["opponent"].tolist())

    changed = before - after
    if changed:
        print(f"  [WCQ] Nomes normalizados: {sorted(changed)}")
    else:
        print("  [WCQ] Nenhuma normalização necessária.")

    # home_advantage: 1 apenas para WCQ home em campo não-neutro
    df["home_advantage"] = (
        (df["home_away"] == "home") &
        (~df["match_id"].isin(NEUTRAL_WCQ_IDS))
    ).astype(int)

    wcq = df[OUTPUT_COLS].copy()
    print(f"  WCQ: {wcq['match_id'].nunique()} jogos, {len(wcq)} linhas")
    return wcq


# ---------------------------------------------------------------------------
# PASSO 1b — Amistosos (FBref, nível de jogador → Squad Total)
# ---------------------------------------------------------------------------

def _parse_score(score: pd.Series, side: pd.Series):
    """Vetorizado: '3–1' + 'home' → gols_marcados=3, gols_sofridos=1."""
    clean = score.astype(str).str.replace(r"[^\d]+", "-", regex=True).str.strip("-")
    parts = clean.str.extract(r"^(\d+)-(\d+)$")
    h = pd.to_numeric(parts[0], errors="coerce")
    a = pd.to_numeric(parts[1], errors="coerce")
    gm = pd.Series(np.where(side == "home", h, a).astype(float), index=score.index)
    gs = pd.Series(np.where(side == "home", a, h).astype(float), index=score.index)
    return gm, gs


def load_friendlies() -> pd.DataFrame:
    df = pd.read_csv(FRIENDLIES_CSV)

    mask = df["player"].astype(str).str.match(r"^\d+ Players?$", na=False)
    sq = df[mask].copy()

    n_total   = df["match_id"].nunique()
    n_sq_mtch = sq["match_id"].nunique()
    print(f"  Friendlies: {n_total} jogos carregados, {n_sq_mtch} com Squad Total rows ({len(sq)} linhas)")

    if sq.empty:
        print("  AVISO: nenhuma linha Squad Total em friendlies — ignorando.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    sq = sq.reset_index(drop=True)

    # Derivar opponent a partir de home_team/away_team
    sq["_team"]     = sq["team_name"].astype(str)
    sq["_opponent"] = np.where(sq["team_side"] == "home", sq["away_team"], sq["home_team"]).astype(str)

    gm, gs = _parse_score(sq["score"], sq["team_side"])

    date_parsed = pd.to_datetime(sq["date"], errors="coerce")

    def _col(name: str) -> pd.Series:
        if name in sq.columns:
            return pd.to_numeric(sq[name], errors="coerce")
        return pd.Series(np.nan, index=sq.index)

    out = pd.DataFrame({
        "match_id":        sq["match_id"].astype(str),
        "date":            sq["date"],
        "competition":     sq["competition"],
        "confederation":   "FRIENDLY",
        "match_type":      "friendly",
        "season":          date_parsed.dt.year,
        "home_away":       sq["team_side"],
        "team":            sq["_team"],
        "opponent":        sq["_opponent"],
        "gols_marcados":   gm,
        "gols_sofridos":   gs,
        "shots":           _col("performance_sh"),
        "shots_on_goal":   _col("performance_sot"),
        "blocked_shots":   np.nan,
        "ball_possession": np.nan,
        "fouls":           _col("performance_fls"),
        "yellow_cards":    _col("performance_crdy"),
        "red_cards":       _col("performance_crdr"),
        "corners":         np.nan,
        "offsides":        _col("performance_off"),
        "passes_accurate": np.nan,
        "saves":           np.nan,
        "ast":             _col("performance_ast"),
        "has_shot_data":   _col("performance_sh").notna(),
        "weight":          1.0,
        "home_advantage":  0,  # amistosos: sem vantagem de campo
    })

    before = set(out["team"].tolist()) | set(out["opponent"].tolist())
    out["team"]     = _normalize(out["team"])
    out["opponent"] = _normalize(out["opponent"])
    after = set(out["team"].tolist()) | set(out["opponent"].tolist())
    changed = before - after
    if changed:
        print(f"  [Friendlies] Nomes normalizados: {sorted(changed)}")
    else:
        print("  [Friendlies] Nenhuma normalização necessária.")

    print(f"  Friendlies Squad Total: {out['match_id'].nunique()} jogos, {len(out)} linhas")
    return out[OUTPUT_COLS]


# ---------------------------------------------------------------------------
# Relatório de cobertura Copa 2026
# ---------------------------------------------------------------------------

def _copa_coverage(df: pd.DataFrame) -> None:
    all_teams = set(df["team"].unique())
    in_data   = COPA2026_TEAMS & all_teams
    missing   = COPA2026_TEAMS - all_teams

    print(f"\n  Times Copa 2026 cobertos: {len(in_data)}/46")
    if missing:
        print(f"  Não encontrados (possível variante de nome):")
        for t in sorted(missing):
            print(f"    '{t}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  integrate_datasets.py — PASSOS 1–2")
    print("=" * 60)

    print("\n── PASSO 1 — WCQ ──")
    wcq = load_wcq()
    # red_cards: null → 0 onde has_shot_data=True no WCQ
    mask_wcq = wcq["has_shot_data"].map(
        {"True": True, "False": False, True: True, False: False}
    ).fillna(False)
    wcq.loc[mask_wcq & wcq["red_cards"].isna(), "red_cards"] = 0.0

    # Extras legado (concacaf_extra_raw.csv, pode não existir)
    extras_frames = []
    if EXTRAS_CSV.exists():
        ex = pd.read_csv(EXTRAS_CSV)
        ex["match_id"] = ex["match_id"].astype(str)
        ex["team"]     = _normalize(ex["team"])
        ex["opponent"] = _normalize(ex["opponent"])
        ex["home_advantage"] = 0
        extras_frames.append(ex[OUTPUT_COLS])
        print(f"\n── CONCACAF extras legado ──")
        print(f"  {EXTRAS_CSV.name}: {ex['match_id'].nunique()} jogos, {len(ex)} linhas")

    print("\n── PASSO 1 — Torneios adicionais (hosts + AFCON) ──")
    tournament_frames = load_tournament_extras()

    print("\n── PASSO 1 — Amistosos ──")
    fr = load_friendlies()

    frames = [wcq] + extras_frames + tournament_frames + [fr]
    all_df = pd.concat(frames, ignore_index=True)

    # Deduplicação: mantém primeira ocorrência de (match_id, team)
    before = len(all_df)
    all_df = all_df.drop_duplicates(subset=["match_id", "team"], keep="first")
    removed = before - len(all_df)
    if removed:
        print(f"\n  Deduplicação: {removed} linhas removidas (match_id + team duplicados)")

    print("\n── Cobertura Copa 2026 (dataset completo) ──")
    _copa_coverage(all_df)

    print(f"\n── PASSO 2 — Salvando {OUTPUT_CSV} ──")
    print(f"  WCQ      : {len(wcq):>5} linhas ({wcq['match_id'].nunique()} jogos)")
    for ef in extras_frames:
        print(f"  CONCACAF+: {len(ef):>5} linhas ({ef['match_id'].nunique()} jogos)")
    print(f"  Friendly : {len(fr):>5} linhas ({fr['match_id'].nunique()} jogos)")
    print(f"  Total    : {len(all_df):>5} linhas ({all_df['match_id'].nunique()} jogos únicos)")

    all_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"  → Salvo: {OUTPUT_CSV}")

    # Distribuição por match_type e confederation
    print("\n  Jogos por match_type:")
    for mt, cnt in all_df.groupby("match_type")["match_id"].nunique().sort_values(ascending=False).items():
        print(f"    {mt:<10}: {cnt}")

    print("\n  Jogos por confederation:")
    for conf, cnt in all_df.groupby("confederation")["match_id"].nunique().sort_values(ascending=False).items():
        shot_pct = all_df[(all_df["confederation"] == conf) & (all_df["has_shot_data"] == True)]["match_id"].nunique()
        total = cnt
        pct = shot_pct / total * 100 if total else 0
        print(f"    {conf:<12}: {cnt:>4} jogos  (shots: {shot_pct}/{total} = {pct:.0f}%)")


if __name__ == "__main__":
    main()
