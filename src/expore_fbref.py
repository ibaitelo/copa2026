"""
Explora dados de qualificatórias CONMEBOL Copa 2026 via soccerdata/FBref.

Fluxo:
  1. Registra "CONMEBOL-WCQ" no league_dict customizado do soccerdata
  2. Navega diretamente à página de stats 2026 e encontra o schedule
     (a página de "history" do qualifying usa <div>, não <table id="seasons">
      — o caminho padrão do soccerdata não funciona para esta competição)
  3. Baixa e parseia a tabela de resultados (Scores & Fixtures)
  4. Tenta também baixar team match logs para checar dados de 1º/2º tempo
  5. Salva CSV bruto em data/raw/conmebol_quali_raw.csv
  6. Imprime resumo no terminal
"""

import io
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Registra liga customizada antes de importar soccerdata
# ---------------------------------------------------------------------------
SOCCERDATA_CONFIG = Path.home() / "soccerdata" / "config"
SOCCERDATA_CONFIG.mkdir(parents=True, exist_ok=True)
LEAGUE_DICT_PATH = SOCCERDATA_CONFIG / "league_dict.json"

LEAGUE_KEY = "CONMEBOL-WCQ"
# Nome exato como aparece na página /en/comps/ do FBref (U+2014 em-dash, ID 4)
FBREF_COMPETITION_NAME = "FIFA World Cup Qualification — CONMEBOL"

existing = {}
if LEAGUE_DICT_PATH.exists():
    with open(LEAGUE_DICT_PATH, encoding="utf-8") as f:
        existing = json.load(f)

existing[LEAGUE_KEY] = {
    "FBref": FBREF_COMPETITION_NAME,
    "season_code": "single-year",
}
with open(LEAGUE_DICT_PATH, "w", encoding="utf-8") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 2. Importa soccerdata e patcha o LEAGUE_DICT em runtime
# ---------------------------------------------------------------------------
import soccerdata as sd  # noqa: E402
import pandas as pd  # noqa: E402
from lxml import html as lxhtml  # noqa: E402

from soccerdata import _config as _sd_config  # noqa: E402
from soccerdata.fbref import (  # noqa: E402
    FBREF_API,
    FBREF_DATADIR,
    _parse_table,
)
from soccerdata._common import standardize_colnames, make_game_id  # noqa: E402
from soccerdata._config import TEAMNAME_REPLACEMENTS  # noqa: E402

_sd_config.LEAGUE_DICT[LEAGUE_KEY] = existing[LEAGUE_KEY]
if hasattr(sd.FBref, "_all_leagues_dict"):
    del sd.FBref._all_leagues_dict

# ---------------------------------------------------------------------------
# 3. Inicializa o leitor
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  FBref — Qualificatórias CONMEBOL Copa do Mundo 2026")
print("=" * 65)

fbref = sd.FBref(
    leagues=LEAGUE_KEY,
    seasons=2026,
    no_cache=False,
    no_store=False,
)

# URLs diretas para a temporada 2026 (sem passar por read_seasons)
# A página de "history" usa <div>, não <table id="seasons">, por isso
# navegamos direto à página da temporada atual.
STATS_URL = f"{FBREF_API}/en/comps/4/WCQ----CONMEBOL-M-Stats"
STATS_CACHE = FBREF_DATADIR / f"teams_{LEAGUE_KEY}_2026.html"

# ---------------------------------------------------------------------------
# 4. Baixa página de stats para achar link "Scores & Fixtures"
# ---------------------------------------------------------------------------
print("\n[1/4] Acessando página de stats 2026...")
try:
    reader_stats = fbref.get(STATS_URL, STATS_CACHE)
    tree_stats = lxhtml.parse(reader_stats)

    scores_links = tree_stats.xpath("//a[text()='Scores & Fixtures']")
    if not scores_links:
        # Tenta texto alternativo em inglês/espanhol que FBref usa
        scores_links = tree_stats.xpath("//a[contains(text(),'Scores') and contains(text(),'Fixtures')]")
    if not scores_links:
        raise RuntimeError("Link 'Scores & Fixtures' não encontrado na página de stats.")

    SCHEDULE_URL = FBREF_API + scores_links[0].get("href")
    print(f"  Link de fixtures encontrado: {SCHEDULE_URL}")
except Exception as exc:
    print(f"  ERRO: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Baixa e parseia a tabela de resultados
# ---------------------------------------------------------------------------
print("\n[2/4] Baixando schedule (resultados / placares)...")
SCHEDULE_CACHE = FBREF_DATADIR / f"schedule_{LEAGUE_KEY}_2026.html"

schedule_df = None
try:
    reader_sched = fbref.get(SCHEDULE_URL, SCHEDULE_CACHE, no_cache=False)
    tree_sched = lxhtml.parse(reader_sched)

    html_tables = tree_sched.xpath("//table[contains(@id, 'sched')]")
    if not html_tables:
        raise RuntimeError("Tabela de schedule (id='sched_...') não encontrada.")

    html_table = html_tables[0]
    df = _parse_table(html_table)

    # Extrai href real do match report (substitui a coluna de texto "Match Report")
    # _parse_table captura o texto "Match Report" mas não o href; precisamos do href.
    mr_hrefs = [
        (
            cell.xpath("./a/@href")[0]
            if cell.xpath("./a") and cell.xpath("./a")[0].text == "Match Report"
            else None
        )
        for cell in html_table.xpath(".//td[@data-stat='match_report']")
    ]
    # Remove coluna de texto gerada por _parse_table (valor = "Match Report" ou vazio)
    for col in ("Match Report", "match_report"):
        if col in df.columns:
            df = df.drop(columns=[col])
    df["match_report"] = mr_hrefs

    df["league"] = LEAGUE_KEY
    df["season"] = "2026"
    df = df.dropna(how="all")

    df = (
        df.rename(columns={
            "Wk": "week",
            "Home": "home_team",
            "Away": "away_team",
            "xG": "home_xg",
            "xG.1": "away_xg",
        })
        .replace({
            "home_team": TEAMNAME_REPLACEMENTS,
            "away_team": TEAMNAME_REPLACEMENTS,
        })
        .pipe(standardize_colnames)
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").ffill()

    schedule_df = df
    print(f"  Linhas no schedule: {len(schedule_df)}")
    print(f"  Colunas: {schedule_df.columns.tolist()}")

except Exception as exc:
    print(f"  ERRO ao parsear schedule: {exc}")

# ---------------------------------------------------------------------------
# 6. Verifica team match logs para dados de 1º/2º tempo
# ---------------------------------------------------------------------------
print("\n[3/4] Verificando team match logs (shooting — inclui métricas por jogo)...")
matchlog_sample_df = None
try:
    # Pega URL de um time a partir da página de stats
    reader_stats2 = fbref.get(STATS_URL, STATS_CACHE)
    tree_stats2 = lxhtml.parse(reader_stats2)

    # Todos os links de squad na página de stats
    team_links = tree_stats2.xpath(
        "//td[@data-stat='team']/a/@href | "
        "//td[@data-stat='squad']/a/@href"
    )
    if not team_links:
        team_links = [
            h for h in tree_stats2.xpath("//a/@href")
            if "/en/squads/" in h and "-Stats" in h
        ][:1]

    if team_links:
        # Monta URL de matchlogs de shooting para o primeiro time
        squad_url = team_links[0]  # e.g. /en/squads/abc123/Team-Men-Stats
        base = squad_url.rsplit("/", 1)[0]
        matchlog_url = FBREF_API + base + "/matchlogs/all_comps/shooting"
        print(f"  Testando matchlog URL: {matchlog_url}")
        # Apenas exibe disponibilidade; não baixa todos para não demorar
        print("  (match logs por time disponíveis via fbref.read_team_match_stats(stat_type='shooting'))")
    else:
        print("  Nenhum link de squad encontrado na página de stats.")

except Exception as exc:
    print(f"  Nota: {exc}")

# ---------------------------------------------------------------------------
# 7. Salva CSV bruto
# ---------------------------------------------------------------------------
output_path = Path("data/raw/conmebol_quali_raw.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

print("\n[4/4] Salvando CSV...")
if schedule_df is not None:
    schedule_df.to_csv(output_path, index=False)
    print(f"  Salvo em: {output_path}  ({len(schedule_df)} linhas)")
else:
    print("  NENHUM dado disponível para salvar.")

# ---------------------------------------------------------------------------
# 8. Resumo no terminal
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  RESUMO")
print("=" * 65)

if schedule_df is None:
    print("\n  Nenhum dado encontrado.")
    sys.exit(1)

df = schedule_df

# Total de linhas
print(f"\n  Total de linhas (jogos + linhas sem placar) : {len(df)}")

# Jogos completos (com placar)
score_col = next(
    (c for c in df.columns if c.lower() in ("score", "resultado", "placar")), None
)
if score_col:
    completed = df[df[score_col].notna() & (df[score_col].astype(str).str.strip() != "")]
    print(f"  Jogos com placar                           : {len(completed)}")

# Times únicos
teams: set = set()
for col in ("home_team", "away_team"):
    if col in df.columns:
        teams.update(df[col].dropna().astype(str).unique())
teams.discard("")
print(f"\n  Times únicos ({len(teams)}):")
for team in sorted(teams):
    print(f"    - {team}")

# Colunas de 1º/2º tempo
ht_cols = [
    c for c in df.columns
    if any(x in c.lower() for x in ("ht", "half", "1st", "2nd", "primeiro", "segundo"))
]
print(f"\n  Colunas com dados de 1º/2º tempo  : ", end="")
if ht_cols:
    print(f"{ht_cols}")
else:
    print("nenhuma detectada no schedule.")
    print("  Dados de half-time disponíveis via:")
    print("    fbref.read_team_match_stats(stat_type='schedule') — col 'ht_score' nos match logs")

# Todas as colunas
print(f"\n  Todas as colunas disponíveis ({len(df.columns)}):")
for col in df.columns:
    print(f"    {col}")

print("\n" + "=" * 65 + "\n")
