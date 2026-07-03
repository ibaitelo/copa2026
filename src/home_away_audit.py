"""
home_away_audit.py — Audita o viés home/away no dataset de treino.

Auditorias:
  1 — Amistosos com venue suspeito (cruza time home vs país do venue)
  2 — Impacto no home_advantage (média de gols por categoria)
  3 — Proporção de jogos afetados (% do treino com home_advantage incorreto)
  4 — Verificação via API (amistosos home do Brasil → venue.city)

Saída: outputs/home_away_audit.txt
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT       = Path(__file__).parent.parent
FULL_RAW   = ROOT / "data/raw/full_dataset_raw.csv"
CACHE_DIR  = ROOT / "data/raw/api_cache/fixtures"
AUDIT_CACHE = ROOT / "data/raw/api_cache/audit_brazil_venues.json"
OUT_TXT    = ROOT / "outputs/home_away_audit.txt"
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

load_dotenv(encoding="utf-8-sig")
API_KEY    = os.getenv("API_FOOTBALL_KEY", "")
API_BASE   = "https://v3.football.api-sports.io"
BRAZIL_ID  = 6    # API-Football team ID (confirmado no cache WCQ)

# ─────────────────────────────────────────────────────────────────────────────
# Mapeamentos
# ─────────────────────────────────────────────────────────────────────────────

# Ligas neutras (todos os jogos são em campo neutro, independente de home/away)
NEUTRAL_LEAGUES: dict[int, str] = {
    9:  "Copa América 2024 (EUA)",
    22: "CONCACAF Gold Cup 2023/2025 (EUA)",
}

# Ligas com home/away genuíno
GENUINE_HOME_LEAGUES: set[int] = {29, 30, 31, 32, 33, 34, 6, 36}  # WCQ + AFCON

# Liga CONCACAF NL: fase regular é home/away real; Finals são neutros
CONCACAF_NL_LEAGUE = 536

# Time → país (48 seleções Copa 2026 + comuns)
TEAM_COUNTRY: dict[str, str] = {
    # CONMEBOL
    "Argentina": "Argentina", "Brazil": "Brazil", "Colombia": "Colombia",
    "Ecuador": "Ecuador", "Uruguay": "Uruguay", "Paraguay": "Paraguay",
    # UEFA
    "Germany": "Germany", "France": "France", "Spain": "Spain",
    "England": "England", "Netherlands": "Netherlands", "Portugal": "Portugal",
    "Belgium": "Belgium", "Austria": "Austria", "Switzerland": "Switzerland",
    "Croatia": "Croatia", "Norway": "Norway", "Sweden": "Sweden",
    "Czech Republic": "Czech Republic", "Turkey": "Turkey",
    "Scotland": "Scotland", "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    # AFC
    "Japan": "Japan", "South Korea": "South Korea", "Iran": "Iran",
    "Australia": "Australia", "Saudi Arabia": "Saudi Arabia",
    "Uzbekistan": "Uzbekistan", "Jordan": "Jordan", "Iraq": "Iraq",
    "Qatar": "Qatar",
    # CAF
    "Morocco": "Morocco", "Senegal": "Senegal", "Egypt": "Egypt",
    "Ghana": "Ghana", "Cape Verde": "Cape Verde", "DR Congo": "DR Congo",
    "Ivory Coast": "Ivory Coast", "South Africa": "South Africa",
    "Algeria": "Algeria", "Tunisia": "Tunisia",
    # CONCACAF
    "United States": "United States", "USA": "United States",
    "Mexico": "Mexico", "Canada": "Canada",
    "Panama": "Panama", "Curacao": "Curacao", "Haiti": "Haiti",
    # OFC
    "New Zealand": "New Zealand",
}

# Palavras-chave no venue.city que indicam país
CITY_COUNTRY_CLUES: list[tuple[list[str], str]] = [
    # Brasil
    (["São Paulo", "Sao Paulo", "Rio de Janeiro", "Belo Horizonte",
      "Recife", "Porto Alegre", "Fortaleza", "Manaus", "Brasília",
      "Brasilia", "Salvador", "Curitiba", "Natal", "Cuiabá"],   "Brazil"),
    # EUA (estados ou cidades icônicas)
    (["New York", "New Jersey", "Texas", "California", "Florida",
      "Illinois", "Arizona", "Georgia", "Nevada", "Massachusetts",
      "Ohio", "New Orleans", "Los Angeles", "Miami", "Chicago",
      "Atlanta", "Houston", "Dallas", "Seattle", "Denver",
      "Las Vegas", "Glendale", "Arlington", "Orlando",
      "East Rutherford", "Inglewood", "Santa Clara", "Harrison",
      "Fort Lauderdale", "San Jose", "St. Louis", "Kansas City",
      "Washington"],                                             "United States"),
    # Argentina
    (["Buenos Aires", "Córdoba", "Rosario", "Mendoza"],         "Argentina"),
    # Alemanha
    (["Berlin", "München", "Munich", "Hamburg", "Frankfurt",
      "Stuttgart", "Dortmund", "Leipzig"],                      "Germany"),
    # França
    (["Paris", "Lyon", "Marseille", "Bordeaux", "Lens", "Nice"], "France"),
    # Espanha
    (["Madrid", "Barcelona", "Seville", "Valencia", "Bilbao",
      "San Sebastián"],                                          "Spain"),
    # Marrocos
    (["Rabat", "Casablanca", "Marrakech", "Fes", "Agadir"],    "Morocco"),
    # Arábia Saudita
    (["Riyadh", "Jeddah", "Dammam"],                            "Saudi Arabia"),
    # Catar
    (["Doha", "Lusail"],                                         "Qatar"),
    # Portugal
    (["Lisbon", "Lisboa", "Porto", "Braga"],                    "Portugal"),
    # Japão
    (["Tokyo", "Osaka", "Yokohama", "Saitama", "Nagoya"],       "Japan"),
]


def city_to_country(city: str) -> str | None:
    """Tenta inferir o país a partir do nome da cidade."""
    if not city:
        return None
    city_lower = city.lower()
    for keywords, country in CITY_COUNTRY_CLUES:
        for kw in keywords:
            if kw.lower() in city_lower:
                return country
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Carga de dados
# ─────────────────────────────────────────────────────────────────────────────

def load_full_dataset() -> pd.DataFrame:
    df = pd.read_csv(FULL_RAW)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["gols_marcados"] = pd.to_numeric(df["gols_marcados"], errors="coerce")
    return df


def load_fixture_cache() -> pd.DataFrame:
    """Lê todos os league*.json do cache e retorna DataFrame com info de venue."""
    rows = []
    for jfile in sorted(CACHE_DIR.glob("league*.json")):
        with open(jfile, encoding="utf-8") as fp:
            d = json.load(fp)
        for fix in d.get("response", []):
            fid    = str(fix.get("fixture", {}).get("id", ""))
            venue  = fix.get("fixture", {}).get("venue", {}) or {}
            home   = fix.get("teams", {}).get("home", {}).get("name", "")
            away   = fix.get("teams", {}).get("away", {}).get("name", "")
            home_g = (fix.get("goals") or {}).get("home")
            away_g = (fix.get("goals") or {}).get("away")
            lid    = fix.get("league", {}).get("id")
            lnm    = fix.get("league", {}).get("name", "")
            date   = fix.get("fixture", {}).get("date", "")[:10]
            rows.append({
                "fixture_id":  fid,
                "home":        home,
                "away":        away,
                "home_goals":  home_g,
                "away_goals":  away_g,
                "league_id":   lid,
                "league":      lnm,
                "date":        date,
                "venue_city":  (venue.get("city") or ""),
                "venue_name":  (venue.get("name") or ""),
            })
    vdf = pd.DataFrame(rows)
    vdf["venue_country_inferred"] = vdf["venue_city"].apply(city_to_country)
    return vdf


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria 1 — Venue cruzamento
# ─────────────────────────────────────────────────────────────────────────────

def auditoria_1(df: pd.DataFrame, vdf: pd.DataFrame, lines: list[str]) -> None:
    lines += [
        "=" * 70,
        "  AUDITORIA 1 — VENUES SUSPEITOS (home team ≠ país do venue)",
        "=" * 70,
        "",
    ]

    # ── 1a. Ligas comprovadamente neutras (Copa América, Gold Cup) ─────────
    lines.append("  1a. Ligas com venue neutro comprovado (Copa América / Gold Cup)")
    lines.append("      Todas as partidas são disputadas nos EUA — nenhum time joga")
    lines.append("      realmente em casa, mesmo que listado como 'home'.")
    lines.append("")

    neutral_rows = vdf[vdf["league_id"].isin(NEUTRAL_LEAGUES)].copy()
    for lid, lname in NEUTRAL_LEAGUES.items():
        sub = neutral_rows[neutral_rows["league_id"] == lid]
        non_usa_home = sub[~sub["home"].isin(["USA", "United States"])]["home"].value_counts()
        lines.append(f"  {lname}")
        lines.append(f"    Total jogos: {len(sub)}")
        lines.append(f"    Times home que NÃO são dos EUA (venue neutro): {len(non_usa_home)} times únicos")
        if not non_usa_home.empty:
            for team, cnt in non_usa_home.head(8).items():
                lines.append(f"      {team:<28}: {cnt} jogo(s) como 'home' em solo americano")
        lines.append("")

    # ── 1b. WCQ — verifica se home team condiz com venue ──────────────────
    lines.append("  1b. WCQ — home team vs país inferido do venue")
    lines.append("")

    wcq_rows = vdf[vdf["league_id"].isin(GENUINE_HOME_LEAGUES)].copy()
    wcq_rows["home_country"]   = wcq_rows["home"].map(TEAM_COUNTRY)
    wcq_rows["venue_country"]  = wcq_rows["venue_country_inferred"]

    # Jogos onde venue_country ≠ home_country (e ambos conhecidos)
    both_known = wcq_rows[
        wcq_rows["home_country"].notna() & wcq_rows["venue_country"].notna()
    ]
    mismatch = both_known[both_known["home_country"] != both_known["venue_country"]]

    lines.append(f"  WCQ total fixtures: {len(wcq_rows)}")
    lines.append(f"  Com venue_city inferível: {len(both_known)} ({len(both_known)/len(wcq_rows)*100:.1f}%)")
    lines.append(f"  Home ≠ venue country (suspeitos): {len(mismatch)}")
    if not mismatch.empty:
        lines.append("  Exemplos:")
        for _, row in mismatch.head(10).iterrows():
            lines.append(f"    {row['date']}  home={row['home']} ({row['home_country']})"
                         f"  venue_city={row['venue_city']} ({row['venue_country']})")
    lines.append("")

    # ── 1c. Amistosos FBref — sem info de venue ────────────────────────────
    lines.append("  1c. Amistosos FBref (63.3% dos amistosos no dataset)")
    lines.append("      Fonte: FBref — não há campo venue/city nessa fonte.")
    lines.append("      Home/away no FBref é atribuído pelo scraper do próprio FBref,")
    lines.append("      que frequentemente lista jogos em campo neutro com designação")
    lines.append("      arbitrária (time 'home' pode estar jogando no exterior).")
    lines.append("      → Impacto quantificado na Auditoria 4.")
    lines.append("")

    # Injetar no dataset principal os dados de venue
    neutral_ids = set(neutral_rows["fixture_id"])
    df["is_confirmed_neutral"] = df["match_id"].isin(neutral_ids)

    n_neutral_rows = int(df["is_confirmed_neutral"].sum())
    lines.append(f"  Total linhas com venue neutro CONFIRMADO no dataset: {n_neutral_rows}")
    lines.append("")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria 2 — Impacto no home_advantage (médias de gols)
# ─────────────────────────────────────────────────────────────────────────────

def auditoria_2(df: pd.DataFrame, lines: list[str]) -> None:
    lines += [
        "=" * 70,
        "  AUDITORIA 2 — IMPACTO NO HOME_ADVANTAGE (médias de gols)",
        "=" * 70,
        "",
    ]

    # Classifica cada linha por categoria
    def classify(row) -> str:
        mt    = row.get("match_type", "")
        haway = row.get("home_away", "")
        mid   = str(row.get("match_id", ""))

        if mt == "WCQ":
            return f"WCQ_{haway.upper()}"

        is_neutral_confirmed = row.get("is_confirmed_neutral", False)
        if is_neutral_confirmed:
            return f"TOURNAMENT_NEUTRAL_{haway.upper()}"

        if mt == "tournament":
            return f"TOURNAMENT_{haway.upper()}"  # CONCACAF NL etc.

        # friendly
        is_numeric = mid.isdigit()
        if is_numeric:
            return f"FRIENDLY_API_{haway.upper()}"
        return f"FRIENDLY_FBREF_{haway.upper()}"

    df["categoria"] = df.apply(classify, axis=1)

    cats_order = [
        "WCQ_HOME", "WCQ_AWAY",
        "TOURNAMENT_NEUTRAL_HOME", "TOURNAMENT_NEUTRAL_AWAY",
        "TOURNAMENT_HOME", "TOURNAMENT_AWAY",
        "FRIENDLY_FBREF_HOME", "FRIENDLY_FBREF_AWAY",
        "FRIENDLY_API_HOME", "FRIENDLY_API_AWAY",
    ]

    lines.append(f"  {'Categoria':<32}  {'N jogos':>8}  {'Média gols':>10}  {'Std':>6}")
    lines.append(f"  {'-'*32}  {'-'*8}  {'-'*10}  {'-'*6}")

    stats: dict[str, dict] = {}
    for cat in cats_order:
        sub = df[df["categoria"] == cat]["gols_marcados"].dropna()
        if len(sub) == 0:
            continue
        mu  = sub.mean()
        std = sub.std()
        lines.append(f"  {cat:<32}  {len(sub):>8}  {mu:>10.3f}  {std:>6.3f}")
        stats[cat] = {"n": len(sub), "mean": mu, "std": std}

    lines.append("")

    # Teste: WCQ home vs away (sinal real de home advantage)
    wcq_h = df[df["categoria"] == "WCQ_HOME"]["gols_marcados"].dropna()
    wcq_a = df[df["categoria"] == "WCQ_AWAY"]["gols_marcados"].dropna()
    if len(wcq_h) > 0 and len(wcq_a) > 0:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(wcq_h, wcq_a, alternative="greater")
        delta = wcq_h.mean() - wcq_a.mean()
        lines.append(f"  WCQ home_adv: home={wcq_h.mean():.3f}  away={wcq_a.mean():.3f}"
                     f"  Δ={delta:+.3f}  Mann-Whitney p={pval:.4f}"
                     + ("  ← home advantage REAL" if pval < 0.05 else "  ← não significativo"))
        lines.append("")

    # Teste: Tournament neutral home vs away (esperado: sem diferença)
    tn_h = df[df["categoria"] == "TOURNAMENT_NEUTRAL_HOME"]["gols_marcados"].dropna()
    tn_a = df[df["categoria"] == "TOURNAMENT_NEUTRAL_AWAY"]["gols_marcados"].dropna()
    if len(tn_h) > 2 and len(tn_a) > 2:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(tn_h, tn_a, alternative="two-sided")
        delta = tn_h.mean() - tn_a.mean()
        lines.append(f"  Copa América/GoldCup home vs away: Δ={delta:+.3f}  p={pval:.4f}"
                     + ("  ← diferença significativa! (inesperado)" if pval < 0.05
                        else "  ← sem diferença (campo neutro confirmado)"))
        lines.append("")

    # Teste: FBref friendly home vs away (esperado: mínimo ou zero)
    fr_h = df[df["categoria"] == "FRIENDLY_FBREF_HOME"]["gols_marcados"].dropna()
    fr_a = df[df["categoria"] == "FRIENDLY_FBREF_AWAY"]["gols_marcados"].dropna()
    if len(fr_h) > 10 and len(fr_a) > 10:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(fr_h, fr_a, alternative="two-sided")
        delta = fr_h.mean() - fr_a.mean()
        bias_level = "ALTO (campo neutro não registrado)" if abs(delta) > 0.2 else \
                     "MODERADO" if abs(delta) > 0.1 else "BAIXO"
        lines.append(f"  FBref friendly home vs away: home={fr_h.mean():.3f}  away={fr_a.mean():.3f}"
                     f"  Δ={delta:+.3f}  p={pval:.4f}")
        lines.append(f"  → Nível de viés: {bias_level}")
        lines.append("")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria 3 — Proporção de jogos afetados
# ─────────────────────────────────────────────────────────────────────────────

def auditoria_3(df: pd.DataFrame, lines: list[str]) -> None:
    lines += [
        "=" * 70,
        "  AUDITORIA 3 — PROPORÇÃO DE JOGOS COM HOME_ADVANTAGE INCORRETO",
        "=" * 70,
        "",
    ]

    total = len(df)
    home_rows = df[df["home_away"] == "home"]
    total_home = len(home_rows)

    # 1. Copa América + Gold Cup home rows = CERTAMENTE errado
    confirmed_neutral_home = df[
        (df["is_confirmed_neutral"] == True) & (df["home_away"] == "home")
    ]
    n_confirmed = len(confirmed_neutral_home)

    # 2. FBref friendly home rows = PROVAVELMENTE errado (campo neutro sem registro)
    fbref_friendly_home = df[
        (df["match_type"] == "friendly")
        & (df["home_away"] == "home")
        & (~df["match_id"].astype(str).str.isdigit())
    ]
    n_fbref = len(fbref_friendly_home)

    # 3. CONCACAF NL home rows (parcialmente suspeito — finals são neutros)
    concacaf_nl_home = df[
        (df["match_type"] == "tournament")
        & (df["home_away"] == "home")
        & (~df["is_confirmed_neutral"])
    ]
    n_concacaf = len(concacaf_nl_home)

    n_total_affected_certain = n_confirmed
    n_total_affected_likely  = n_confirmed + n_fbref
    n_total_affected_possible = n_confirmed + n_fbref + n_concacaf

    lines.append(f"  Dataset de treino (total linhas): {total}")
    lines.append(f"  Linhas com home_away='home':      {total_home}")
    lines.append("")
    lines.append("  Casos onde home_advantage=1 é INCORRETO ou SUSPEITO:")
    lines.append("")
    lines.append(f"  [CERTAMENTE INCORRETO] Copa América / Gold Cup home:")
    lines.append(f"    N = {n_confirmed}  ({n_confirmed/total*100:.1f}% do dataset total)")
    lines.append(f"    Motivo: todos os jogos em solo americano (campo neutro)")
    lines.append("")
    lines.append(f"  [PROVAVELMENTE INCORRETO] Amistosos FBref home:")
    lines.append(f"    N = {n_fbref}  ({n_fbref/total*100:.1f}% do dataset total)")
    lines.append(f"    Motivo: FBref não distingue home real de campo neutro;")
    lines.append(f"    a Auditoria 4 quantifica isso para o Brasil especificamente")
    lines.append("")
    lines.append(f"  [PARCIALMENTE SUSPEITO] CONCACAF NL home (excl. Copa América/GC):")
    lines.append(f"    N = {n_concacaf}  ({n_concacaf/total*100:.1f}% do dataset total)")
    lines.append(f"    Motivo: fases de grupo são home/away reais; Finals são neutros")
    lines.append("")

    pct_certain  = n_total_affected_certain  / total * 100
    pct_likely   = n_total_affected_likely   / total * 100
    pct_possible = n_total_affected_possible / total * 100

    lines.append(f"  ┌─ Estimativa de viés total ──────────────────────────────────┐")
    lines.append(f"  │  Mínimo (certamente incorreto):    {pct_certain:>5.1f}% do dataset   │")
    lines.append(f"  │  Provável (+ amistosos FBref):     {pct_likely:>5.1f}% do dataset   │")
    lines.append(f"  │  Possível (+ CONCACAF NL):         {pct_possible:>5.1f}% do dataset   │")
    lines.append(f"  └─────────────────────────────────────────────────────────────┘")
    lines.append("")

    threshold = 20.0
    if pct_likely >= threshold:
        verdict = f"MATERIAL (≥{threshold}%) — correção necessária"
    elif pct_likely >= 10.0:
        verdict = "MODERADO (10-20%) — correção recomendada"
    else:
        verdict = f"BAIXO (<10%) — impacto limitado"
    lines.append(f"  VEREDICTO AUDITORIA 3: viés é {verdict}")
    lines.append("")

    return n_confirmed, n_fbref


# ─────────────────────────────────────────────────────────────────────────────
# Auditoria 4 — Verificação via API (Brasil)
# ─────────────────────────────────────────────────────────────────────────────

def _api_get(endpoint: str, params: dict) -> dict:
    """Chama a API-Football com headers corretos."""
    headers = {"x-apisports-key": API_KEY}
    url     = f"{API_BASE}{endpoint}"
    resp    = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _load_audit_cache() -> dict:
    if AUDIT_CACHE.exists():
        with open(AUDIT_CACHE, encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def _save_audit_cache(data: dict) -> None:
    AUDIT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_CACHE, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def auditoria_4(df: pd.DataFrame, lines: list[str]) -> None:
    lines += [
        "=" * 70,
        "  AUDITORIA 4 — VERIFICAÇÃO VIA API (amistosos home do Brasil)",
        "=" * 70,
        "",
    ]

    if not API_KEY:
        lines.append("  ERRO: API_FOOTBALL_KEY não encontrada em .env — auditoria ignorada.")
        lines.append("")
        return

    br_df = df[df["team"] == "Brazil"].sort_values("date").copy()
    br_home_friendly = br_df[
        (br_df["home_away"] == "home") & (br_df["match_type"] == "friendly")
    ]

    lines.append(f"  Brasil no dataset: {len(br_df)} linhas totais")
    lines.append(f"  Brasil home + friendly: {len(br_home_friendly)} linhas")
    lines.append("")
    lines.append(f"  → Busca via API: GET /fixtures?team={BRAZIL_ID}&league=10&season={{ano}}")
    lines.append(f"    (league=10 = Amistosos de Seleções Nacionais na API-Football)")
    lines.append("")

    audit_cache = _load_audit_cache()
    results: list[dict] = []
    new_calls = 0

    for season in [2025, 2026]:
        cache_key = f"brazil_friendly_league10_{season}"
        if cache_key in audit_cache:
            api_data = audit_cache[cache_key]
        else:
            try:
                api_data = _api_get("/fixtures", {
                    "team":   BRAZIL_ID,
                    "league": 10,
                    "season": season,
                    "status": "FT",
                })
                audit_cache[cache_key] = api_data
                new_calls += 1
                time.sleep(0.5)
            except Exception as e:
                lines.append(f"  ERRO ao buscar season {season}: {e}")
                continue

        for fix in api_data.get("response", []):
            fid       = fix.get("fixture", {}).get("id")
            venue     = fix.get("fixture", {}).get("venue", {}) or {}
            v_city    = venue.get("city", "") or ""
            v_name    = venue.get("name", "") or ""
            home_name = fix.get("teams", {}).get("home", {}).get("name", "")
            away_name = fix.get("teams", {}).get("away", {}).get("name", "")
            date_str  = fix.get("fixture", {}).get("date", "")[:10]

            br_side   = "home" if "Brazil" in home_name else "away"
            opp       = away_name if br_side == "home" else home_name

            v_country    = city_to_country(v_city)
            is_in_brazil = (v_country == "Brazil")
            venue_status = "✓ EM CASA (Brasil)" if is_in_brazil else \
                           ("✗ NEUTRO (" + (v_country or v_city or "?") + ")")

            results.append({
                "date": date_str, "br_side": br_side, "opponent": opp,
                "venue_city": v_city, "venue_name": v_name,
                "venue_status": venue_status, "is_in_brazil": is_in_brazil,
                "season": season,
            })

    if new_calls > 0:
        _save_audit_cache(audit_cache)

    if not results:
        lines.append("  Nenhum dado retornado pela API.")
        lines.append("")
        return

    home_results = [r for r in results if r["br_side"] == "home"]
    n_genuinely_home = sum(1 for r in home_results if r["is_in_brazil"])
    n_neutral        = len(home_results) - n_genuinely_home
    pct_neutral      = n_neutral / len(home_results) * 100 if home_results else 0

    lines.append(f"  Amistosos encontrados (2025+2026): {len(results)}")
    lines.append(f"  Brasil listado como 'home': {len(home_results)}")
    lines.append("")
    lines.append(f"  {'Data':<12}  {'Adversário':<22}  {'Venue (Cidade)':<28}  Status")
    lines.append(f"  {'-'*12}  {'-'*22}  {'-'*28}  {'-'*30}")
    for r in sorted(home_results, key=lambda x: x["date"]):
        lines.append(f"  {r['date']:<12}  {r['opponent']:<22}  "
                     f"{r['venue_city']:<28}  {r['venue_status']}")
    lines.append("")
    lines.append(f"  EM CASA (Rio/SP/etc.): {n_genuinely_home} / {len(home_results)}"
                 f"  ({100-pct_neutral:.0f}%)")
    lines.append(f"  CAMPO NEUTRO         : {n_neutral} / {len(home_results)}"
                 f"  ({pct_neutral:.0f}%)")
    lines.append("")
    if pct_neutral >= 50:
        lines.append("  → CONFIRMADO: maioria dos amistosos 'home' do Brasil jogados")
        lines.append("    em solo estrangeiro (Europa, EUA). O rótulo home_advantage=1")
        lines.append("    nesses jogos é INCORRETO. Generaliza para as demais seleções.")
    else:
        lines.append("  → Maioria dos amistosos 'home' do Brasil jogados EM CASA.")
    lines.append("")


# ─────────────────────────────────────────────────────────────────────────────
# Conclusão
# ─────────────────────────────────────────────────────────────────────────────

def conclusao(df: pd.DataFrame, n_confirmed: int, n_fbref: int,
               lines: list[str]) -> None:
    lines += [
        "=" * 70,
        "  CONCLUSÃO INTEGRADA",
        "=" * 70,
        "",
    ]
    total = len(df)
    total_home = len(df[df["home_away"] == "home"])
    pct_min  = n_confirmed / total * 100
    pct_prov = (n_confirmed + n_fbref) / total * 100

    lines.append("  Fontes de viés identificadas:")
    lines.append("")
    lines.append(f"  1. COPA AMÉRICAR + GOLD CUP (campo neutro confirmado)")
    lines.append(f"     {n_confirmed} linhas home com home_advantage=1 INCORRETO"
                 f" ({pct_min:.1f}% do dataset)")
    lines.append("")
    lines.append(f"  2. AMISTOSOS FBref (campo neutro provável)")
    lines.append(f"     {n_fbref} linhas home sem informação de venue"
                 f" ({n_fbref/total*100:.1f}% do dataset)")
    lines.append(f"     FBref atribui home/away com base na listagem da partida,")
    lines.append(f"     não na localização geográfica do estádio.")
    lines.append("")
    lines.append(f"  3. IMPACTO TOTAL ESTIMADO NO TREINO")
    lines.append(f"     Mínimo (certamente): {pct_min:.1f}% | Provável: {pct_prov:.1f}%")
    lines.append("")

    is_material = pct_prov >= 20.0

    lines.append("  ┌─────────────────────────────────────────────────────────────┐")
    if is_material:
        lines.append("  │  ★ VIÉS É MATERIAL — CORREÇÃO NECESSÁRIA ANTES DO TREINO ★  │")
    else:
        lines.append("  │    Viés moderado — correção recomendada mas não crítica       │")
    lines.append("  └─────────────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append("  AÇÃO RECOMENDADA:")
    lines.append("    • Copa América / Gold Cup: definir home_advantage=0 (campo neutro)")
    lines.append("    • Amistosos FBref: definir home_advantage=0 por padrão")
    lines.append("      (conservative approach — campo neutro é o mais comum)")
    lines.append("    • WCQ: manter home_advantage=1 (correto por definição)")
    lines.append("    • CONCACAF NL fase de grupos: manter home_advantage=1")
    lines.append("    • Na simulação Copa 2026: home_advantage=0 (sede neutro)")
    lines.append("")
    lines.append("  IMPLEMENTAÇÃO SUGERIDA em integrate_datasets.py:")
    lines.append("    Adicionar coluna 'home_advantage_raw' no pipeline:")
    lines.append("      WCQ       → home_advantage_raw = 1 (se home_away=='home')")
    lines.append("      tournament Copa América/GoldCup → 0 (sempre campo neutro)")
    lines.append("      friendly  → 0 (campo neutro por padrão)")
    lines.append("    Manter home_advantage=1 apenas para WCQ e CONCACAF NL regular")
    lines.append("")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  home_away_audit.py — Auditoria de viés home/away")
    print("=" * 70)

    print("  Carregando dados ...")
    df  = load_full_dataset()
    vdf = load_fixture_cache()
    print(f"  Dataset: {len(df)} linhas | Fixture cache: {len(vdf)} registros")

    lines: list[str] = []
    lines += [
        "=" * 70,
        "  HOME/AWAY BIAS AUDIT — Copa 2026 prediction project",
        f"  Gerado em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Dataset  : {FULL_RAW.name}  ({len(df)} linhas, {df['match_id'].nunique()} jogos)",
        "=" * 70,
        "",
    ]

    print("  Auditoria 1 ...")
    df = auditoria_1(df, vdf, lines)

    print("  Auditoria 2 ...")
    df = auditoria_2(df, lines)

    print("  Auditoria 3 ...")
    n_conf, n_fbref = auditoria_3(df, lines)

    print("  Auditoria 4 (API) ...")
    auditoria_4(df, lines)

    conclusao(df, n_conf, n_fbref, lines)

    report = "\n".join(lines)
    print(report)

    with open(OUT_TXT, "w", encoding="utf-8") as fp:
        fp.write(report + "\n")

    print(f"\n  → Salvo: {OUT_TXT}")


if __name__ == "__main__":
    main()
