"""
Normaliza data/raw/all_confederations_raw.csv e salva em
data/processed/all_confederations_clean.csv.

Correções aplicadas:
  1. Remove códigos ISO embutidos nos nomes de times
  2. Remove linhas exatamente duplicadas (bug de scraping CAF)
  3. Adiciona flag has_shot_data
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAW_CSV   = Path("data/raw/all_confederations_raw.csv")
OUT_CSV   = Path("data/processed/all_confederations_clean.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Times classificados para Copa 2026 (nomes FBref)
# ---------------------------------------------------------------------------
COPA2026_TEAMS = [
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
    # Playoffs inter-confederações (2 — ajustar quando confirmados)
]

STAT_COLS_SHORT = ["sh", "sot", "gls", "ast", "tklw", "int", "fls", "crdy", "crdr"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ISO_PREFIX = re.compile(r"^[a-z]{2,3} ")
_ISO_SUFFIX = re.compile(r" [a-z]{2,3}$")


def strip_iso(name: str) -> str:
    """Remove prefixo/sufixo de 2-3 letras de código de país/FIFA
    ('hr Croatia' → 'Croatia', 'eng England' → 'England')."""
    s = str(name).strip()
    s = _ISO_PREFIX.sub("", s)
    s = _ISO_SUFFIX.sub("", s)
    return s


def _section(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  NORMALIZE DATA — Copa 2026")
    print("=" * 65)

    df = pd.read_csv(RAW_CSV, low_memory=False)
    rows_before = len(df)
    teams_before = set()
    for col in ("home_team", "away_team", "team_name"):
        if col in df.columns:
            teams_before.update(df[col].dropna().astype(str).unique())

    print(f"\n  Dataset original: {rows_before:,} linhas, "
          f"{df['match_id'].nunique() if 'match_id' in df.columns else '?'} jogos únicos, "
          f"{len(teams_before)} nomes de times")

    # -----------------------------------------------------------------------
    # CORREÇÃO 1 — Normalizar nomes de times
    # -----------------------------------------------------------------------
    _section("CORREÇÃO 1 — Normalizar nomes de times (remover códigos ISO)")

    team_cols = [c for c in ("home_team", "away_team", "team_name") if c in df.columns]
    changed = 0
    for col in team_cols:
        original = df[col].copy()
        df[col] = df[col].apply(lambda v: strip_iso(v) if pd.notna(v) else v)
        changed += (original != df[col]).sum()

    print(f"  Valores alterados: {changed:,} em {team_cols}")

    teams_after = set()
    for col in team_cols:
        teams_after.update(df[col].dropna().astype(str).unique())
    print(f"  Nomes únicos: {len(teams_before)} → {len(teams_after)}")

    # Top 20 nomes mais frequentes para validação
    name_freq = df["team_name"].value_counts().head(20) if "team_name" in df.columns else pd.Series()
    print(f"\n  Top 20 team_name mais frequentes:")
    for name, cnt in name_freq.items():
        print(f"    {cnt:>6}  {name}")

    # Conferir se ainda existem ISO residuais
    iso_left = [n for n in teams_after
                if _ISO_PREFIX.match(str(n)) or _ISO_SUFFIX.search(str(n))]
    if iso_left:
        print(f"\n  ⚠  ISO residuais ({len(iso_left)}): {iso_left[:10]}")
    else:
        print("\n  ✓  Nenhum código ISO residual.")

    # -----------------------------------------------------------------------
    # CORREÇÃO 2 — Remover duplicatas
    # -----------------------------------------------------------------------
    _section("CORREÇÃO 2 — Remover linhas duplicadas")

    # Chave correta: inclui player porque o dado é nível de jogador
    # (match_id + team_name + stat_type sozinhos identificam ~14 jogadores por grupo)
    dup_key = [c for c in ("match_id", "team_name", "stat_type", "player")
               if c in df.columns]

    dups_mask = df.duplicated(subset=dup_key, keep="first")
    n_dups = dups_mask.sum()

    if n_dups > 0:
        dup_rows = df[dups_mask]
        print(f"  Linhas duplicadas encontradas: {n_dups}")
        print(f"  Match_ids afetados: {dup_rows['match_id'].nunique()}")
        print(f"\n  Por competição:")
        if "competition" in dup_rows.columns:
            by_comp = dup_rows.groupby("competition").size().sort_values(ascending=False)
            for comp, cnt in by_comp.items():
                print(f"    {cnt:>4}  {comp}")
        df = df[~dups_mask].reset_index(drop=True)
        print(f"\n  Linhas após remoção: {len(df):,}")
    else:
        print("  Nenhuma duplicata exata encontrada.")

    # Verificar duplicatas de jogo (mesmo home/away/date após normalização)
    if all(c in df.columns for c in ("home_team", "away_team", "date", "match_id")):
        sched = df.drop_duplicates("match_id")[["match_id", "home_team", "away_team", "date"]]
        game_dups = sched[sched.duplicated(subset=["home_team", "away_team", "date"], keep=False)]
        if len(game_dups) > 0:
            print(f"\n  ⚠  Jogos duplicados (mesmo home/away/date, match_id diferente): "
                  f"{game_dups['match_id'].nunique()}")
            # Para cada grupo, manter o match_id com mais sh não-nulo
            sh_col = next((c for c in df.columns if c.endswith("_sh")), None)
            game_dups_merged = game_dups.merge(
                df.groupby("match_id")[sh_col].count().rename("sh_count").reset_index()
                if sh_col else pd.DataFrame(columns=["match_id", "sh_count"]),
                on="match_id", how="left"
            )
            keep_ids = set()
            for _, group in game_dups_merged.groupby(["home_team", "away_team", "date"]):
                best = group.sort_values("sh_count", ascending=False).iloc[0]["match_id"]
                keep_ids.add(best)
            drop_ids = set(game_dups["match_id"]) - keep_ids
            df = df[~df["match_id"].isin(drop_ids)].reset_index(drop=True)
            print(f"  Removidos {len(drop_ids)} match_ids duplicados de jogo.")
        else:
            print("  ✓  Nenhum jogo duplicado (home/away/date) após normalização.")

    # -----------------------------------------------------------------------
    # CORREÇÃO 3 — Flag has_shot_data
    # -----------------------------------------------------------------------
    _section("CORREÇÃO 3 — Criar flag has_shot_data")

    sh_col = next((c for c in df.columns if c.endswith("_sh") or c == "sh"), None)
    if sh_col:
        df["has_shot_data"] = df[sh_col].notna()
        print(f"  Flag criada com base em: {sh_col!r}")
        print(f"\n  Distribuição por confederação:")
        if "confederation" in df.columns:
            dist = df.groupby("confederation")["has_shot_data"].agg(
                total="count",
                com_shot="sum",
            )
            dist["pct"] = (dist["com_shot"] / dist["total"] * 100).round(1)
            print(dist.to_string())
        total_true  = df["has_shot_data"].sum()
        total_false = (~df["has_shot_data"]).sum()
        print(f"\n  Total com shot data:    {total_true:>8,} ({total_true/len(df)*100:.1f}%)")
        print(f"  Total sem shot data:    {total_false:>8,} ({total_false/len(df)*100:.1f}%)")
    else:
        df["has_shot_data"] = False
        print("  ⚠  Coluna sh não encontrada — flag = False para todas as linhas.")

    # -----------------------------------------------------------------------
    # Salvar
    # -----------------------------------------------------------------------
    _section("SALVANDO")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    rows_after = len(df)
    teams_final = set()
    for col in team_cols:
        teams_final.update(df[col].dropna().astype(str).unique())

    print(f"\n  → {OUT_CSV}")
    print(f"\n  Resumo:")
    print(f"    Linhas antes:        {rows_before:>8,}")
    print(f"    Linhas depois:       {rows_after:>8,}  (removidas: {rows_before - rows_after:,})")
    print(f"    Times únicos antes:  {len(teams_before):>8,}")
    print(f"    Times únicos depois: {len(teams_final):>8,}")

    # Times Copa 2026 não cobertos
    import difflib
    not_found = []
    fuzzy_maps = {}
    for t in COPA2026_TEAMS:
        if t in teams_final:
            continue
        close = difflib.get_close_matches(t, teams_final, n=1, cutoff=0.72)
        if close:
            fuzzy_maps[t] = close[0]
        else:
            not_found.append(t)

    print(f"\n  Times Copa 2026 cobertos: {len(COPA2026_TEAMS) - len(not_found)}/{len(COPA2026_TEAMS)}")
    if not_found:
        print(f"  ⚠  NÃO ENCONTRADOS ({len(not_found)}):")
        for t in not_found:
            print(f"       - {t!r}")
    if fuzzy_maps:
        print(f"  Mapeamentos aproximados (verificar nome no FBref):")
        for copa, dataset_name in sorted(fuzzy_maps.items()):
            print(f"    Copa: {copa!r}  →  Dataset: {dataset_name!r}")


if __name__ == "__main__":
    main()
