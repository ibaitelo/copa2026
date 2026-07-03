"""
Coleta match stats do FBref para todas as confederações da Copa do Mundo 2026.

Competições cobertas:
  UEFA:     Nations League 2024/25 (703), Euro 2024 (676), WCQ 2026 (682)
  CONCACAF: WCQ 2026 (3), Gold Cup 2023 (686)
  AFC:      WCQ 2026 (36), Asian Cup 2024 (687)
  CAF:      WCQ 2026 (78), AFCON 2023 (689)
  CONMEBOL: Copa América 2024 (685)
  OFC:      WCQ 2026 (91)

Retomável: cache de HTMLs + pula match_ids já presentes no CSV de saída.
Rate-limit: 3 s entre requisições reais (HTTPs com cache são instantâneas).

Saídas:
  data/raw/{CONFEDERATION}_raw.csv   — uma por confederação
  data/raw/all_confederations_raw.csv — tudo concatenado (inclui CONMEBOL WCQ existente)
"""

import io
import json
import re
import sys
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows consoles (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from lxml import html as lxhtml
from lxml import etree

# ---------------------------------------------------------------------------
# Soccerdata / FBref client setup (espelha fetch_match_stats.py)
# ---------------------------------------------------------------------------
SOCCERDATA_CONFIG = Path.home() / "soccerdata" / "config"
SOCCERDATA_CONFIG.mkdir(parents=True, exist_ok=True)
LEAGUE_DICT_PATH = SOCCERDATA_CONFIG / "league_dict.json"

_DUMMY_KEY = "CONMEBOL-WCQ"
_DUMMY_NAME = "FIFA World Cup Qualification — CONMEBOL"

_leagues: dict = {}
if LEAGUE_DICT_PATH.exists():
    with open(LEAGUE_DICT_PATH, encoding="utf-8") as _f:
        _leagues = json.load(_f)
_leagues[_DUMMY_KEY] = {"FBref": _DUMMY_NAME, "season_code": "single-year"}
with open(LEAGUE_DICT_PATH, "w", encoding="utf-8") as _f:
    json.dump(_leagues, _f, indent=2, ensure_ascii=False)

import soccerdata as sd  # noqa: E402
from soccerdata import _config as _sd_config  # noqa: E402
from soccerdata.fbref import FBREF_API, FBREF_DATADIR  # noqa: E402

_sd_config.LEAGUE_DICT[_DUMMY_KEY] = _leagues[_DUMMY_KEY]
if hasattr(sd.FBref, "_all_leagues_dict"):
    del sd.FBref._all_leagues_dict

# ---------------------------------------------------------------------------
# Competition definitions
# ---------------------------------------------------------------------------
# hints: FBref path candidates (relative to FBREF_API), tried in order.
# Pattern for WCQ: /en/comps/{id}/WCQ----{CONF}-M-Stats  (mirrors CONMEBOL WCQ comp 4)
# Pattern for tournaments: /en/comps/{id}/{year}/{Slug}-Stats
# Each hint list should go from most-specific to most-generic.
COMPETITIONS = [
    # UEFA
    dict(key="UEFA-NL",   name="UEFA Nations League 2024/25", comp_id=677,
         confederation="UEFA",     weight=0.8,
         hints=[
             "/en/comps/677/2024-2025/schedule/2024-25-UEFA-Nations-League-Scores-and-Fixtures",
             "/en/comps/677/2024-2025/Nations-League-Stats",
             "/en/comps/677/",
         ]),
    dict(key="UEFA-EURO", name="UEFA Euro 2024",               comp_id=676,
         confederation="UEFA",     weight=1.0,
         hints=[
             "/en/comps/676/2024/schedule/2024-European-Championship-Scores-and-Fixtures",
             "/en/comps/676/2024/European-Championship-Stats",
             "/en/comps/676/European-Championship-Stats",
         ]),
    dict(key="UEFA-WCQ",  name="UEFA WCQ 2026",                comp_id=6,
         confederation="UEFA",     weight=1.0,
         hints=[
             "/en/comps/6/WCQ----UEFA-M-Stats",
         ]),
    # CONCACAF
    dict(key="CONCACAF-WCQ", name="CONCACAF WCQ 2026",         comp_id=3,
         confederation="CONCACAF", weight=1.0,
         hints=[
             "/en/comps/3/WCQ----CONCACAF-M-Stats",
             "/en/comps/3/2026/schedule/",
         ]),
    dict(key="CONCACAF-GC",  name="CONCACAF Gold Cup 2023",    comp_id=681,
         confederation="CONCACAF", weight=1.0,
         hints=[
             "/en/comps/681/2023/schedule/2023-Gold-Cup-Scores-and-Fixtures",
             "/en/comps/681/2023/Gold-Cup-Stats",
             "/en/comps/681/Gold-Cup-Stats",
         ]),
    # AFC
    dict(key="AFC-WCQ", name="AFC WCQ 2026",                   comp_id=7,
         confederation="AFC",      weight=1.0,
         hints=[
             "/en/comps/7/WCQ----AFC-M-Stats",
         ]),
    dict(key="AFC-AC",  name="AFC Asian Cup 2024",              comp_id=664,
         confederation="AFC",      weight=1.0,
         hints=[
             "/en/comps/664/2024/schedule/2024-AFC-Asian-Cup-Scores-and-Fixtures",
             "/en/comps/664/2024/AFC-Asian-Cup-Stats",
             "/en/comps/664/AFC-Asian-Cup-Stats",
         ]),
    # CAF
    dict(key="CAF-WCQ",   name="CAF WCQ 2026",                 comp_id=2,
         confederation="CAF",      weight=1.0,
         hints=[
             "/en/comps/2/schedule/WCQ----CAF-M-Scores-and-Fixtures",
             "/en/comps/2/WCQ----CAF-M-Stats",
         ]),
    dict(key="CAF-AFCON", name="AFCON 2023",                   comp_id=656,
         confederation="CAF",      weight=1.0,
         # AFCON "2023" was held in January-February 2024
         hints=[
             "/en/comps/656/2024/schedule/2023-Africa-Cup-of-Nations-Scores-and-Fixtures",
             "/en/comps/656/2023/schedule/2023-Africa-Cup-of-Nations-Scores-and-Fixtures",
             "/en/comps/656/2024/Africa-Cup-of-Nations-Stats",
             "/en/comps/656/Africa-Cup-of-Nations-Stats",
         ]),
    # CONMEBOL
    dict(key="CONMEBOL-CA", name="Copa América 2024",           comp_id=685,
         confederation="CONMEBOL", weight=1.0,
         hints=[
             "/en/comps/685/2024/schedule/2024-Copa-America-Scores-and-Fixtures",
             "/en/comps/685/2024/Copa-America-Stats",
             "/en/comps/685/Copa-America-Stats",
         ]),
    # OFC
    dict(key="OFC-WCQ", name="OFC WCQ 2026",                   comp_id=5,
         confederation="OFC",      weight=1.0,
         hints=[
             "/en/comps/5/WCQ----OFC-M-Stats",
         ]),
]

# Spots per confederation at Copa 2026 (direct + playoff, approximate)
COPA2026_SPOTS = {
    "UEFA": 16, "CAF": 9, "CONMEBOL": 6,
    "AFC": 8, "CONCACAF": 6, "OFC": 1,
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STAT_TYPES = ["summary"]
START_DATE = pd.Timestamp("2023-01-01")

RAW_DIR = Path("data/raw")
COMP_CACHE   = FBREF_DATADIR / "comp_pages"
SCHED_CACHE  = FBREF_DATADIR / "schedules"
MATCH_CACHE  = FBREF_DATADIR / "match_reports"

for _d in (RAW_DIR, COMP_CACHE, SCHED_CACHE, MATCH_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

CONMEBOL_EXISTING = RAW_DIR / "conmebol_match_stats_raw.csv"
ALL_OUT = RAW_DIR / "all_confederations_raw.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_match_id(href: str) -> str:
    m = re.search(r"/en/matches/([a-f0-9]+)/", str(href))
    return m.group(1) if m else ""


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


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"[\s%]+", "_", c.lower().strip()).strip("_") for c in df.columns]
    return df


def _fetch_html(client, url: str, cache_path: Path) -> Optional[str]:
    """Fetch URL (with cache) and return decoded HTML string, or None on error."""
    try:
        reader = client.get(url, cache_path)
        raw = reader.read()
        return raw.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"      HTTP error [{url}]: {e}")
        return None


def parse_sched_element(table_el) -> pd.DataFrame:
    """
    Parse a FBref schedule <table> element row-by-row.
    Extracts data-stat values + match_report hrefs directly from the DOM,
    avoiding MultiIndex column alignment issues from pd.read_html.
    """
    rows = []
    for tr in table_el.xpath(".//tbody/tr"):
        cls = tr.get("class", "")
        if any(x in cls for x in ("spacer", "partial_table", "thead")):
            continue
        row: dict = {}
        for cell in tr.xpath("./td|./th"):
            stat = cell.get("data-stat", "")
            if not stat:
                continue
            if stat == "match_report":
                hrefs = cell.xpath("./a/@href")
                row["match_report"] = hrefs[0] if hrefs else None
            else:
                row[stat] = cell.text_content().strip()
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


_HOMEPAGE_MARKERS = ("Premier-League-Stats", "id=\"header_clubs\"", "id=\"header_players\"")


def _is_invalid_page(html: str) -> bool:
    """Return True if the HTML is not a valid competition/schedule page.

    Rejects:
      - FBref homepage (soft-404 for completely wrong URLs)
      - Generic "Football Competitions" index (returned for invalid comp slugs)
      - Daily "Football Matches on..." page (returned for /en/matches/)

    Uses <title> (always in first ~2 KB) as the primary signal since <h1>
    can be buried thousands of bytes into the document.
    """
    # Search <title> in first 3000 chars (always in <head>)
    head3k = html[:3000]
    title_m = re.search(r"<title>([^<]+)</title>", head3k, re.IGNORECASE)
    if title_m:
        title = title_m.group(1).strip()
        if title.startswith("Football Competitions"):
            return True
        if title.startswith("Football Matches on"):
            return True

    # Also check first 5000 chars for FBref homepage markers (soft-404)
    head5k = html[:5000]
    if sum(1 for m in _HOMEPAGE_MARKERS if m in head5k) >= 2:
        return True

    # Fallback: search deeper for h1 (some FBref pages have delayed body)
    head20k = html[:20000]
    if re.search(r"<h1[^>]*>\s*Football Competitions\s*<", head20k):
        return True
    if re.search(r"<h1[^>]*>\s*Football Matches on\b", head20k):
        return True

    return False


def find_schedule_url(client, comp: dict) -> Optional[str]:
    """
    Tries to discover the Scores & Fixtures URL for a FBref competition.

    Strategy:
      1. Try each hint path in comp['hints']  (most specific first)
      2. For each candidate, if the page IS a schedule table → return that URL
         If the page has a 'Scores & Fixtures' link → follow it
         If the page is the FBref homepage (soft-404) → delete cache, skip
    """
    comp_id = comp["comp_id"]
    candidates = [FBREF_API + h for h in comp.get("hints", [])]

    for url in candidates:
        slug = re.sub(r"[^a-z0-9]", "_", url.lower())[-60:]
        cache_path = COMP_CACHE / f"nav_{comp_id}_{slug}.html"

        # Pre-check: delete cache if it previously stored invalid content
        if cache_path.exists():
            try:
                peek = cache_path.read_bytes()[:5000].decode("utf-8", errors="replace")
                if _is_invalid_page(peek):
                    cache_path.unlink()
            except Exception:
                pass

        html = _fetch_html(client, url, cache_path)
        if html is None:
            continue

        if _is_invalid_page(html):
            print(f"      [{url[-60:]}] pagina invalida (homepage/generico) — ignorando")
            try:
                cache_path.unlink()
            except Exception:
                pass
            continue

        html_clean = html.replace("<!--", "").replace("-->", "")
        tree = lxhtml.fromstring(html_clean)

        # Page IS the schedule
        if tree.xpath("//table[contains(@id, 'sched_')]"):
            return url

        # Page has a 'Scores & Fixtures' link that belongs to THIS competition
        for a_el in tree.xpath("//a[contains(text(), 'Scores') and contains(text(), 'Fixtures')]"):
            href = a_el.get("href", "")
            if href and href.startswith("/") and f"/comps/{comp_id}/" in href:
                result = FBREF_API + href
                print(f"      Scores & Fixtures: {result[-80:]}")
                return result

        # Any link on the page that points to this comp's schedule
        sched_links = [
            a.get("href", "")
            for a in tree.xpath(
                f"//a[contains(@href, '/comps/{comp_id}') and contains(@href, '/schedule')]"
            )
            if a.get("href", "").startswith("/")
        ]
        if sched_links:
            result = FBREF_API + sched_links[0]
            print(f"      Schedule link: {result[-80:]}")
            return result

        # Last resort: if URL ends with "-Stats", derive the schedule path
        # e.g. /en/comps/703/2024-2025/Nations-League-Stats → /en/comps/703/2024-2025/schedule/
        if url.endswith("Stats") or "Stats" in url.split("/")[-1]:
            parts = url.rstrip("/").rsplit("/", 1)
            derived = parts[0] + "/schedule/"
            print(f"      Derivado do Stats URL: {derived[-80:]}")
            return derived

        print(f"      [{url[-60:]}] pagina valida mas sem schedule/fixtures")

    return None


def fetch_schedule(client, schedule_url: str, cache_path: Path) -> Optional[pd.DataFrame]:
    """
    Fetch a schedule (Scores & Fixtures) page and return a cleaned DataFrame.
    Includes only completed matches (have a match_report href) from >= START_DATE.
    """
    html = _fetch_html(client, schedule_url, cache_path)
    if html is None:
        return None

    if _is_invalid_page(html):
        print(f"      AVISO: schedule URL retornou pagina invalida — descartando cache")
        try:
            cache_path.unlink()
        except Exception:
            pass
        return None

    html_clean = html.replace("<!--", "").replace("-->", "")
    tree = lxhtml.fromstring(html_clean)

    tables = tree.xpath("//table[contains(@id, 'sched_')]")
    if not tables:
        print(f"      Nenhuma tabela sched_ encontrada em {schedule_url}")
        return None

    dfs = [parse_sched_element(t) for t in tables]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if df.empty:
        return None

    # Normalize column names (data-stat values are already lowercase but let's be safe)
    df.columns = [re.sub(r"[\s%]+", "_", c.lower().strip()).strip("_") for c in df.columns]

    # Parse and filter by date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] >= START_DATE].copy()
    else:
        print(f"      Aviso: coluna 'date' não encontrada no schedule.")

    # Keep only completed matches (those with a valid match_report href)
    if "match_report" in df.columns:
        df = df[
            df["match_report"].notna()
            & (df["match_report"].astype(str).str.startswith("/en/matches/"))
        ].copy()
    else:
        return None

    # Extract match_id from match_report href
    df["match_id"] = df["match_report"].apply(extract_match_id)
    df = df[df["match_id"] != ""].copy()

    # FBref sometimes has duplicate sched_ tables (home/away perspectives) → deduplicate
    df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    return df


def parse_stat_tables(tree, match_meta: dict, comp: dict) -> list[dict]:
    """
    Extract player/squad stats from a match report HTML tree.
    Adds competition metadata columns before returning.
    """
    rows = []
    for stat_type in STAT_TYPES:
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
            df = norm_cols(df)
            df = df.dropna(how="all")

            if "player" in df.columns:
                df = df[~df["player"].astype(str).str.lower().isin(["player", "nan", ""])]

            # Match context
            df["match_id"]    = match_meta["match_id"]
            df["date"]        = match_meta["date"]
            df["home_team"]   = match_meta["home_team"]
            df["away_team"]   = match_meta["away_team"]
            df["score"]       = match_meta["score"]
            df["team_side"]   = team_side
            df["team_name"]   = team_name
            df["squad_id"]    = squad_id
            df["stat_type"]   = stat_type

            # Competition metadata
            df["competition"]   = comp["name"]
            df["confederation"] = comp["confederation"]
            df["weight"]        = comp["weight"]
            df["year"] = (
                pd.to_datetime(match_meta["date"], errors="coerce").year
                if match_meta["date"] else None
            )

            rows.extend(df.to_dict(orient="records"))

    return rows


def load_done_ids(csv_path: Path) -> set:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            return set(
                pd.read_csv(csv_path, usecols=["match_id"])["match_id"]
                .astype(str).unique()
            )
        except Exception:
            pass
    return set()


# ---------------------------------------------------------------------------
# Per-competition processing
# ---------------------------------------------------------------------------

def process_competition(client, comp: dict) -> int:
    """Download and save stats for one competition. Returns success count."""
    key   = comp["key"]
    cid   = comp["comp_id"]
    conf  = comp["confederation"]
    out   = RAW_DIR / f"{conf}_raw.csv"

    print(f"\n{'-'*65}")
    print(f"  [{key}] {comp['name']}  (comp_id={cid}, w={comp['weight']})")
    print(f"{'-'*65}")

    # 1. Discover schedule URL
    sched_url = find_schedule_url(client, comp)
    if not sched_url:
        print(f"  AVISO: URL de schedule não encontrada. Pulando.")
        return 0
    print(f"  Schedule: {sched_url}")

    # 2. Fetch and parse schedule
    sched_cache = SCHED_CACHE / f"sched_{key}.html"
    schedule = fetch_schedule(client, sched_url, sched_cache)
    if schedule is None or schedule.empty:
        print(f"  AVISO: schedule vazio/não parseável. Pulando.")
        return 0

    print(f"  Jogos completados (>= 2023): {len(schedule)}")

    # 3. Resume: skip already-processed match_ids
    done = load_done_ids(out)
    if done:
        print(f"  Já processados: {len(done)} match_ids — pulando.")

    pending = schedule[~schedule["match_id"].isin(done)].reset_index(drop=True)
    print(f"  A processar: {len(pending)} jogos")

    if pending.empty:
        print(f"  Nada a fazer.")
        return 0

    write_header = not out.exists() or out.stat().st_size == 0
    success = errors = 0

    for i, row in pending.iterrows():
        match_id = row["match_id"]
        href = str(row.get("match_report", "")).strip()
        if not href or href == "nan":
            errors += 1
            continue

        url = (FBREF_API + href) if href.startswith("/") else href
        cache_path = MATCH_CACHE / f"{match_id}.html"

        home = str(row.get("home_team", row.get("home", "?"))).strip()
        away = str(row.get("away_team", row.get("away", "?"))).strip()
        date_val = row.get("date", "")

        print(
            f"  [{i+1}/{len(pending)}] {date_val}  "
            f"{home} vs {away}  ({match_id})"
        )

        # 4. Download match report (cached)
        html = _fetch_html(client, url, cache_path)
        if html is None:
            errors += 1
            continue

        html = html.replace("<!--", "").replace("-->", "")
        tree = lxhtml.fromstring(html)

        match_meta = {
            "match_id":  match_id,
            "date":      str(date_val),
            "home_team": home,
            "away_team": away,
            "score":     str(row.get("score", "")),
        }

        stat_rows = parse_stat_tables(tree, match_meta, comp)

        if not stat_rows:
            print(f"    AVISO: nenhuma tabela extraída.")
            errors += 1
        else:
            pd.DataFrame(stat_rows).to_csv(
                out, mode="a", index=False, header=write_header, encoding="utf-8"
            )
            write_header = False
            print(f"    OK — {len(stat_rows)} linhas")
            success += 1

    print(f"  Concluído: {success} ok, {errors} erros.")
    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = sd.FBref(
        leagues=_DUMMY_KEY,
        seasons=2026,
        no_cache=False,
        no_store=False,
    )
    client.rate_limit = 3

    print("=" * 65)
    print("  FBref — Todas as Confederações Copa do Mundo 2026")
    print("=" * 65)

    for comp in COMPETITIONS:
        process_competition(client, comp)

    # -------------------------------------------------------------------------
    # Concatenate: existing CONMEBOL WCQ + all confederation CSVs
    # -------------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  Concatenando tudo em all_confederations_raw.csv ...")
    print(f"{'='*65}\n")

    dfs: list[pd.DataFrame] = []

    if CONMEBOL_EXISTING.exists():
        df_ex = pd.read_csv(CONMEBOL_EXISTING)
        df_ex["competition"]   = "CONMEBOL WCQ 2026"
        df_ex["confederation"] = "CONMEBOL"
        df_ex["weight"]        = 1.0
        if "date" in df_ex.columns:
            df_ex["year"] = pd.to_datetime(df_ex["date"], errors="coerce").dt.year
        dfs.append(df_ex)
        print(f"  CONMEBOL WCQ (existente): {len(df_ex):>6} linhas")

    confs_seen: set = set()
    for comp in COMPETITIONS:
        conf = comp["confederation"]
        if conf in confs_seen:
            continue
        confs_seen.add(conf)
        csv_path = RAW_DIR / f"{conf}_raw.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"  {conf:<12}: {len(df):>6} linhas")
        except Exception as e:
            print(f"  {conf:<12}: erro ao ler CSV — {e}")

    if not dfs:
        print("  AVISO: nenhum dado para concatenar.")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(ALL_OUT, index=False, encoding="utf-8")
    print(f"\n  → {ALL_OUT}  ({len(all_df):,} linhas totais)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  RESUMO")
    print(f"{'='*65}")

    print("\n  Jogos únicos por confederação:")
    if "match_id" in all_df.columns and "confederation" in all_df.columns:
        by_conf = (
            all_df.groupby("confederation")["match_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        for conf, cnt in by_conf.items():
            print(f"    {conf:<12}: {cnt:>4} jogos")
        print(f"    {'TOTAL':<12}: {all_df['match_id'].nunique():>4} jogos")

    teams: set = set()
    for col in ("home_team", "away_team", "team_name"):
        if col in all_df.columns:
            teams.update(
                t for t in all_df[col].dropna().astype(str).unique()
                if t not in ("", "nan", "?")
            )
    print(f"\n  Times únicos cobertos: {len(teams)}")

    print(f"\n  Cobertura estimada (Copa 2026 — 48 vagas):")
    total_covered = 0
    for conf in sorted(COPA2026_SPOTS):
        spots = COPA2026_SPOTS[conf]
        if "team_name" in all_df.columns and "confederation" in all_df.columns:
            conf_teams = {
                t for t in
                all_df[all_df["confederation"] == conf]["team_name"]
                .dropna().astype(str).unique()
                if t not in ("", "nan", "?")
            }
            covered = min(len(conf_teams), spots)
            total_covered += covered
            pct = covered / spots * 100 if spots else 0
            print(f"    {conf:<12}: {covered:>2}/{spots:<2} times ({pct:.0f}%)")
    print(f"    {'TOTAL':<12}: {total_covered:>2}/48 vagas estimadas")
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
