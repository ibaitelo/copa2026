"""
Baixa tabelas de stats de cada match report FBref das qualificatórias CONMEBOL 2026.

Para cada jogo em data/raw/conmebol_quali_raw.csv:
  - Acessa https://fbref.com/en/matches/<id>/... via soccerdata (Selenium)
  - Extrai tabela "summary" por time (única disponível para CONMEBOL WCQ no FBref)
    Colunas: Player, Pos, Age, Min, Gls, Ast, PK, PKatt, Sh, SoT,
             CrdY, CrdR, Fls, Fld, Off, Crs, TklW, Int, OG, PKwon, PKcon
  - Inclui linha "Squad Total" (útil para features de equipe)
  - Salva tudo em data/raw/conmebol_match_stats_raw.csv

Rate-limit: 3 s entre requests (soccerdata default é 7 s).
Retomável: pula match_ids já presentes no CSV de saída.
"""

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
from lxml import html as lxhtml

# ---------------------------------------------------------------------------
# Registra liga customizada antes de importar soccerdata (igual ao expore_fbref.py)
# ---------------------------------------------------------------------------
SOCCERDATA_CONFIG = Path.home() / "soccerdata" / "config"
SOCCERDATA_CONFIG.mkdir(parents=True, exist_ok=True)
LEAGUE_DICT_PATH = SOCCERDATA_CONFIG / "league_dict.json"

LEAGUE_KEY = "CONMEBOL-WCQ"
FBREF_COMPETITION_NAME = "FIFA World Cup Qualification — CONMEBOL"

existing = {}
if LEAGUE_DICT_PATH.exists():
    with open(LEAGUE_DICT_PATH, encoding="utf-8") as f:
        existing = json.load(f)

existing[LEAGUE_KEY] = {"FBref": FBREF_COMPETITION_NAME, "season_code": "single-year"}
with open(LEAGUE_DICT_PATH, "w", encoding="utf-8") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

import soccerdata as sd  # noqa: E402
from soccerdata import _config as _sd_config  # noqa: E402
from soccerdata.fbref import FBREF_API, FBREF_DATADIR  # noqa: E402

_sd_config.LEAGUE_DICT[LEAGUE_KEY] = existing[LEAGUE_KEY]
if hasattr(sd.FBref, "_all_leagues_dict"):
    del sd.FBref._all_leagues_dict

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
INPUT_CSV = Path("data/raw/conmebol_quali_raw.csv")
OUTPUT_CSV = Path("data/raw/conmebol_match_stats_raw.csv")
MATCH_CACHE_DIR = FBREF_DATADIR / "match_reports"
MATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# FBref disponibiliza apenas "summary" para partidas CONMEBOL WCQ.
# (passing, defense, possession etc. só existem em ligas top-tier no FBref)
STAT_TYPES = ["summary"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_match_id(href: str) -> str:
    m = re.search(r"/en/matches/([a-f0-9]+)/", str(href))
    return m.group(1) if m else str(href)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa MultiIndex de colunas para string simples."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(
                str(c).strip()
                for c in col
                if str(c).strip() and not str(c).startswith("Unnamed")
            ).strip("_")
            for col in df.columns
        ]
    return df


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"[\s\%]+", "_", c.lower().strip()).strip("_") for c in df.columns]
    return df


def parse_stat_tables(tree, match_meta: dict) -> list[dict]:
    """
    Extrai todas as tabelas stats_*_{stat_type} do match report.
    Retorna lista de dicts (uma linha por jogador / squad-total).
    """
    rows = []

    # Obtém string HTML para pd.read_html
    from lxml import etree

    for stat_type in STAT_TYPES:
        # XPath para achar tabelas pelo id
        xpath = f"//table[re:match(@id, 'stats_[a-f0-9]+_{stat_type}$')]"
        tables = tree.xpath(
            xpath,
            namespaces={"re": "http://exslt.org/regular-expressions"},
        )

        for side_idx, table in enumerate(tables):
            team_side = "home" if side_idx == 0 else "away"
            team_name = match_meta["home_team"] if side_idx == 0 else match_meta["away_team"]

            squad_id_m = re.search(r"stats_([a-f0-9]+)_", table.get("id", ""))
            squad_id = squad_id_m.group(1) if squad_id_m else None

            table_html = etree.tostring(table, encoding="unicode")
            try:
                df = pd.read_html(io.StringIO(table_html), header=[0, 1])[0]
            except Exception:
                try:
                    df = pd.read_html(io.StringIO(table_html))[0]
                except Exception as e:
                    print(f"      Não parseable: {stat_type} ({team_side}): {e}")
                    continue

            df = flatten_columns(df)
            df = df.loc[:, ~df.columns.duplicated()]
            df = normalize_colnames(df)
            df = df.dropna(how="all")

            # Remove cabeçalhos repetidos que FBref insere a cada N linhas.
            # Não filtramos pela coluna "#" porque o Squad Total tem # = NaN.
            if "player" in df.columns:
                df = df[~df["player"].astype(str).str.lower().isin(["player", "nan", ""])]

            # Contexto do jogo
            df.insert(0, "match_id", match_meta["match_id"])
            df.insert(1, "date", match_meta["date"])
            df.insert(2, "home_team", match_meta["home_team"])
            df.insert(3, "away_team", match_meta["away_team"])
            df.insert(4, "score", match_meta["score"])
            df.insert(5, "team_side", team_side)
            df.insert(6, "team_name", team_name)
            df.insert(7, "squad_id", squad_id)
            df.insert(8, "stat_type", stat_type)

            rows.extend(df.to_dict(orient="records"))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_CSV.exists():
        print(f"CSV de entrada não encontrado: {INPUT_CSV}")
        sys.exit(1)

    schedule = pd.read_csv(INPUT_CSV)
    schedule["match_id"] = schedule["match_report"].apply(extract_match_id)
    schedule = schedule[
        schedule["score"].notna()
        & schedule["match_report"].notna()
        & (schedule["match_report"].astype(str).str.strip() != "")
    ].copy()
    print(f"Total de jogos no CSV: {len(schedule)}")

    # Retomabilidade — pula match_ids já no CSV de saída
    already_done: set[str] = set()
    if OUTPUT_CSV.exists():
        try:
            existing_out = pd.read_csv(OUTPUT_CSV, usecols=["match_id"])
            already_done = set(existing_out["match_id"].astype(str).unique())
            print(f"Já processados: {len(already_done)} match_ids — serão pulados.")
        except Exception:
            pass

    pending = schedule[~schedule["match_id"].isin(already_done)].reset_index(drop=True)
    print(f"A processar: {len(pending)} jogos\n")

    if pending.empty:
        print("Nada a fazer — todos os jogos já foram baixados.")
        return

    # Inicializa FBref e ajusta rate-limit para 3 s (default é 7 s)
    fbref = sd.FBref(
        leagues=LEAGUE_KEY,
        seasons=2026,
        no_cache=False,
        no_store=False,
    )
    fbref.rate_limit = 3

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    success = 0
    errors = 0

    for i, row in pending.iterrows():
        match_id = row["match_id"]
        href = str(row["match_report"]).strip()
        url = FBREF_API + href
        cache_path = MATCH_CACHE_DIR / f"{match_id}.html"

        print(
            f"[{i + 1}/{len(pending)}] {row.get('date', '')}  "
            f"{row.get('home_team', '')} vs {row.get('away_team', '')}  "
            f"({match_id})"
        )

        try:
            reader = fbref.get(url, cache_path)
            html_bytes = reader.read()
            # FBref esconde tabelas em comentários HTML
            html_str = html_bytes.decode("utf-8", errors="replace")
            html_str = html_str.replace("<!--", "").replace("-->", "")
            tree = lxhtml.fromstring(html_str)
        except Exception as e:
            print(f"  ERRO ao baixar/parsear: {e}\n")
            errors += 1
            continue

        match_meta = {
            "match_id": match_id,
            "date": row.get("date"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "score": row.get("score"),
        }

        rows = parse_stat_tables(tree, match_meta)

        if not rows:
            print(f"  AVISO: nenhuma tabela extraída.\n")
            errors += 1
        else:
            df_out = pd.DataFrame(rows)
            df_out.to_csv(
                OUTPUT_CSV,
                mode="a",
                index=False,
                header=write_header,
                encoding="utf-8",
            )
            write_header = False
            n_types = df_out["stat_type"].nunique()
            print(f"  OK — {len(rows)} linhas / {n_types} tipos de stat\n")
            success += 1

    print("=" * 65)
    print(f"  Concluído: {success} ok, {errors} erros.")
    print(f"  Saída: {OUTPUT_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    main()
