"""
fetch_friendlies.py — Coleta amistosos internacionais masculinos (2025 e 2026) do FBref.

Fluxo:
  1. Acessa as páginas de schedule de 2025 e 2026 da competição 218 (Friendlies-M)
  2. Extrai data, home_team, away_team, score, match_report_href de cada jogo completo
  3. Para cada match report, baixa via soccerdata (Selenium) e parseia tabela "summary"
  4. Adiciona colunas: competition='friendly', weight=0.7, year
  5. Salva em data/raw/friendlies_raw.csv

Rate-limit : 3 s entre requests.
Retomável  : pula match_ids já presentes no CSV de saída.
Cache      : HTMLs salvos em ~\\soccerdata\\data\\FBref\\match_reports\\{match_id}.html

Limitação conhecida do FBref para amistosos:
  - performance_gls, performance_ast, performance_crdy, performance_crdr: disponíveis
  - performance_sh, performance_sot, performance_fls, performance_tklw,
    performance_int: NaN para todos os jogadores e squad total
  O schema é idêntico ao conmebol_match_stats_raw.csv; colunas ausentes ficam NaN.
"""

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
from lxml import html as lxhtml
from lxml import etree

# ---------------------------------------------------------------------------
# Boot soccerdata (idêntico ao fetch_match_stats.py)
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
OUTPUT_CSV = Path("data/raw/friendlies_raw.csv")
MATCH_CACHE_DIR = FBREF_DATADIR / "match_reports"
MATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# URLs de schedule por ano (comp 218 = International Friendlies M)
SCHEDULE_URLS = {
    2025: FBREF_API + "/en/comps/218/2025/schedule/2025-Friendlies-M-Scores-and-Fixtures",
    2026: FBREF_API + "/en/comps/218/schedule/Friendlies-M-Scores-and-Fixtures",
}

COMPETITION = "friendly"
WEIGHT = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_match_id(href: str) -> str:
    m = re.search(r"/en/matches/([a-f0-9]+)/", str(href))
    return m.group(1) if m else str(href)


def parse_schedule_page(tree, year: int) -> pd.DataFrame:
    """
    Extrai linhas do schedule: date, home_team, away_team, score, match_report_href.
    Retorna apenas jogos com match report disponível (partidas concluídas).
    """
    table = tree.xpath("//table[contains(@id,'sched')]")
    if not table:
        print(f"  AVISO: tabela sched não encontrada para {year}")
        return pd.DataFrame()
    table = table[0]

    rows = []
    # Itera pelas linhas de dados (tr com classe even/odd, não thead)
    for tr in table.xpath(".//tbody/tr"):
        # Pula separadores de grupo
        if "spacer" in tr.get("class", "") or "thead" in tr.get("class", ""):
            continue

        def cell(stat: str) -> str:
            c = tr.xpath(f"./td[@data-stat='{stat}']")
            return c[0].text_content().strip() if c else ""

        def cell_a(stat: str) -> str:
            """Texto do <a> dentro da célula, ou cell() como fallback."""
            c = tr.xpath(f"./td[@data-stat='{stat}']/a/text()")
            return c[0].strip() if c else cell(stat)

        def cell_href(stat: str) -> str | None:
            hrefs = tr.xpath(f"./td[@data-stat='{stat}']/a/@href")
            return hrefs[0] if hrefs else None

        # Data: atributo csk="YYYYMMDD" ou texto "YYYY-MM-DD"
        date_cell = tr.xpath("./td[@data-stat='date']")
        if not date_cell:
            continue
        raw_date = date_cell[0].get("csk") or date_cell[0].text_content().strip()
        if not raw_date or not re.match(r"\d{4}", raw_date):
            continue

        mr_href = cell_href("match_report")
        if not mr_href or "/en/matches/" not in mr_href:
            continue  # jogo ainda não realizado ou sem dados

        score = cell("score")
        if not score or not re.search(r"\d", score):
            continue

        rows.append({
            "date": raw_date[:4] + "-" + raw_date[4:6] + "-" + raw_date[6:8]
            if len(raw_date) == 8 and raw_date.isdigit()
            else raw_date,
            "home_team": cell_a("home_team"),
            "away_team": cell_a("away_team"),
            "score": score,
            "match_report": mr_href,
            "year": year,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["match_id"] = df["match_report"].apply(extract_match_id)
    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def parse_summary_tables(tree, match_meta: dict) -> list[dict]:
    """Extrai tabelas stats_*_summary (home + away) do match report."""
    rows = []
    xpath = "//table[re:match(@id, 'stats_[a-f0-9]+_summary$')]"
    tables = tree.xpath(xpath, namespaces={"re": "http://exslt.org/regular-expressions"})

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
                print(f"      parse error ({team_side}): {e}")
                continue

        df = flatten_columns(df)
        df = df.loc[:, ~df.columns.duplicated()]
        df = normalize_colnames(df)
        df = df.dropna(how="all")

        if "player" in df.columns:
            # Remove cabeçalhos repetidos que FBref insere a cada N linhas.
            # Não filtramos pela coluna "#" porque o Squad Total tem # = NaN.
            df = df[~df["player"].astype(str).str.lower().isin(["player", "nan", ""])]

        # Contexto
        df.insert(0, "match_id", match_meta["match_id"])
        df.insert(1, "date", match_meta["date"])
        df.insert(2, "home_team", match_meta["home_team"])
        df.insert(3, "away_team", match_meta["away_team"])
        df.insert(4, "score", match_meta["score"])
        df.insert(5, "team_side", team_side)
        df.insert(6, "team_name", team_name)
        df.insert(7, "squad_id", squad_id)
        df.insert(8, "stat_type", "summary")
        df.insert(9, "competition", match_meta["competition"])
        df.insert(10, "weight", match_meta["weight"])
        df.insert(11, "year", match_meta["year"])

        rows.extend(df.to_dict(orient="records"))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  fetch_friendlies.py — amistosos internacionais 2025/2026")
    print("=" * 65 + "\n")

    fbref = sd.FBref(leagues=LEAGUE_KEY, seasons=2026, no_cache=False, no_store=False)
    fbref.rate_limit = 3

    # ------------------------------------------------------------------
    # FASE 1 — Coleta schedules de 2025 e 2026
    # ------------------------------------------------------------------
    all_games: list[pd.DataFrame] = []

    for year, url in SCHEDULE_URLS.items():
        cache_path = FBREF_DATADIR / f"friendlies_{year}_schedule.html"
        print(f"[Schedule {year}] {url}")
        try:
            reader = fbref.get(url, cache_path)
            html_str = reader.read().decode("utf-8", errors="replace")
            html_str = html_str.replace("<!--", "").replace("-->", "")
            tree = lxhtml.fromstring(html_str)
        except Exception as e:
            print(f"  ERRO ao baixar schedule {year}: {e}")
            continue

        df_year = parse_schedule_page(tree, year)
        print(f"  Jogos com match report em {year}: {len(df_year)}")
        all_games.append(df_year)

    if not all_games:
        print("Nenhum jogo encontrado. Abortando.")
        sys.exit(1)

    schedule = pd.concat(all_games, ignore_index=True).drop_duplicates("match_id")
    print(f"\nTotal de jogos (2025+2026): {len(schedule)}")
    for yr, grp in schedule.groupby("year"):
        print(f"  {yr}: {len(grp)} jogos")

    # ------------------------------------------------------------------
    # FASE 2 — Retomabilidade
    # ------------------------------------------------------------------
    already_done: set[str] = set()
    if OUTPUT_CSV.exists():
        try:
            existing_out = pd.read_csv(OUTPUT_CSV, usecols=["match_id"])
            already_done = set(existing_out["match_id"].astype(str).unique())
            print(f"\nJá processados: {len(already_done)} match_ids — serão pulados.")
        except Exception:
            pass

    pending = schedule[~schedule["match_id"].isin(already_done)].reset_index(drop=True)
    print(f"A processar: {len(pending)} jogos\n")

    if pending.empty:
        print("Nada a fazer — todos os jogos já foram baixados.")
        _print_summary()
        return

    # ------------------------------------------------------------------
    # FASE 3 — Baixa e parseia match reports
    # ------------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    success = 0
    errors = 0

    for i, row in pending.iterrows():
        match_id = row["match_id"]
        url = FBREF_API + str(row["match_report"]).strip()
        cache_path = MATCH_CACHE_DIR / f"{match_id}.html"

        print(
            f"[{i + 1}/{len(pending)}] {row['date']}  "
            f"{row['home_team']} vs {row['away_team']}  ({match_id})"
        )

        try:
            reader = fbref.get(url, cache_path)
            html_str = reader.read().decode("utf-8", errors="replace")
            html_str = html_str.replace("<!--", "").replace("-->", "")
            tree = lxhtml.fromstring(html_str)
        except Exception as e:
            print(f"  ERRO download: {e}\n")
            errors += 1
            continue

        match_meta = {
            "match_id": match_id,
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "score": row["score"],
            "competition": COMPETITION,
            "weight": WEIGHT,
            "year": row["year"],
        }

        rows = parse_summary_tables(tree, match_meta)

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
            print(f"  OK — {len(rows)} linhas\n")
            success += 1

    # ------------------------------------------------------------------
    # FASE 4 — Relatório final
    # ------------------------------------------------------------------
    print("=" * 65)
    print(f"  Concluído: {success} jogos ok, {errors} erros.")
    _print_summary()
    print("=" * 65)


def _print_summary() -> None:
    if not OUTPUT_CSV.exists():
        return
    try:
        df = pd.read_csv(OUTPUT_CSV)
        # Filtra apenas Squad Total para contagens
        squad = df[df["player"].astype(str).str.match(r"^\d+ Players?$", na=False)].copy()
        matches = squad.drop_duplicates("match_id")

        print(f"\n  Jogos no CSV de saída           : {matches['match_id'].nunique()}")
        if "year" in matches.columns:
            for yr, grp in matches.groupby("year"):
                print(f"    {yr}: {len(grp)} jogos")

        teams: set[str] = set()
        for col in ("home_team", "away_team"):
            if col in matches.columns:
                teams.update(matches[col].dropna().unique())
        print(f"\n  Times únicos cobertos           : {len(teams)}")
        for t in sorted(teams):
            print(f"    {t}")

        print(f"\n  Saída: {OUTPUT_CSV}")
    except Exception as e:
        print(f"  (erro ao ler sumário: {e})")


if __name__ == "__main__":
    main()
