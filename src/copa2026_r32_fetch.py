#!/usr/bin/env python3
"""
copa2026_r32_fetch.py — Busca resultados reais do R32 (e rodadas seguintes) via API.

1. Busca todos os fixtures FT da Copa 2026 (league=1 season=2026)
2. Filtra jogos de fase eliminatória (Round of 32, QF, SF, Final) já finalizados
3. Adiciona ao full_dataset_raw.csv (match_type="WC_KO")
4. Imprime chaveamento real para confirmar com o usuário
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(encoding="utf-8-sig")

RAW_CSV  = Path("data/raw/full_dataset_raw.csv")
CACHE_DIR = Path("data/raw/api_cache")

API_KEY  = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}

NAME_MAP: dict[str, str] = {
    "USA":                    "United States",
    "México":                 "Mexico",
    "Curaçao":                "Curacao",
    "Côte d'Ivoire":          "Ivory Coast",
    "Korea Republic":         "South Korea",
    "IR Iran":                "Iran",
    "Bosnia & Herzegovina":   "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":     "Bosnia and Herzegovina",
    "Türkiye":                "Turkey",
    "Cape Verde Islands":     "Cape Verde",
    "Congo DR":               "DR Congo",
    "Czechia":                "Czech Republic",
    "Czech Rep.":             "Czech Republic",
    "South Korea":            "South Korea",
    "Côte d'Ivoire":          "Ivory Coast",
}

COPA_TEAM_CONF: dict[str, str] = {
    "Argentina":"CONMEBOL","Brazil":"CONMEBOL","Colombia":"CONMEBOL",
    "Ecuador":"CONMEBOL","Uruguay":"CONMEBOL","Paraguay":"CONMEBOL",
    "Germany":"UEFA","France":"UEFA","Spain":"UEFA","England":"UEFA",
    "Netherlands":"UEFA","Portugal":"UEFA","Belgium":"UEFA","Austria":"UEFA",
    "Switzerland":"UEFA","Croatia":"UEFA","Norway":"UEFA","Sweden":"UEFA",
    "Czech Republic":"UEFA","Turkey":"UEFA","Scotland":"UEFA",
    "Bosnia and Herzegovina":"UEFA",
    "Japan":"AFC","South Korea":"AFC","Iran":"AFC","Australia":"AFC",
    "Saudi Arabia":"AFC","Uzbekistan":"AFC","Jordan":"AFC","Iraq":"AFC",
    "Qatar":"AFC",
    "Morocco":"CAF","Senegal":"CAF","Egypt":"CAF","Ghana":"CAF",
    "Cape Verde":"CAF","DR Congo":"CAF","Ivory Coast":"CAF",
    "South Africa":"CAF","Algeria":"CAF","Tunisia":"CAF",
    "United States":"CONCACAF","Mexico":"CONCACAF","Canada":"CONCACAF",
    "Panama":"CONCACAF","Curacao":"CONCACAF","Haiti":"CONCACAF",
    "New Zealand":"OFC",
}

COPA2026_TEAM_SET = set(COPA_TEAM_CONF.keys())

STAT_MAP: dict[str, str] = {
    "Total Shots":      "shots",
    "Shots on Goal":    "shots_on_goal",
    "Blocked Shots":    "blocked_shots",
    "Ball Possession":  "ball_possession",
    "Fouls":            "fouls",
    "Yellow Cards":     "yellow_cards",
    "Red Cards":        "red_cards",
    "Corner Kicks":     "corners",
    "Offsides":         "offsides",
    "Passes accurate":  "passes_accurate",
    "Goalkeeper Saves": "saves",
}

# Rounds that are knockout stage (R32 / Round of 32 / 16th finals / etc.)
KNOCKOUT_KEYWORDS = [
    "round of 32", "16th", "1/16", "round of 16", "eighth",
    "quarter", "semi", "final", "third"
]


def _cache_path(endpoint: str, cache_id: str) -> Path:
    p = CACHE_DIR / endpoint.strip("/") / f"{cache_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def api_get_fresh(endpoint: str, params: dict, cache_id: str) -> dict:
    """Busca sem cache (sempre fresh) e salva em cache."""
    url = f"{BASE_URL}/{endpoint.strip('/')}"
    print(f"  API GET {endpoint} params={params}", flush=True)
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30, verify=False)
    resp.raise_for_status()
    data = resp.json()
    cp = _cache_path(endpoint, cache_id)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    time.sleep(0.5)
    return data


def api_get(endpoint: str, params: dict, cache_id: str) -> dict:
    """Com cache."""
    cp = _cache_path(endpoint, cache_id)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            return json.load(f)
    return api_get_fresh(endpoint, params, cache_id)


def _parse_val(val):
    if val is None or val == "null" or val == "":
        return None
    if isinstance(val, str):
        s = val.rstrip("%").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return val
    return val


def fetch_all_ko_games() -> list[dict]:
    """Busca TODOS os jogos Copa 2026 finalizados (status=FT), incluindo fase de grupos e KO."""
    # Fresh fetch — sem cache
    data = api_get_fresh(
        "fixtures",
        {"league": 1, "season": 2026, "status": "FT"},
        cache_id="wc2026_season2026_FT_fresh",
    )
    fixtures = data.get("response", [])
    print(f"  Total fixtures FT retornados pela API: {len(fixtures)}")

    # Identificar rodadas disponíveis
    rounds_found = sorted(set(f.get("league", {}).get("round", "") for f in fixtures))
    print("  Rodadas encontradas:")
    for r in rounds_found:
        cnt = sum(1 for f in fixtures if f.get("league", {}).get("round", "") == r)
        print(f"    [{cnt:>2} jogos] {r}")

    # Filtrar apenas knockout
    ko_fixtures = []
    gs_fixtures = []
    for f in fixtures:
        rnd = f.get("league", {}).get("round", "").lower()
        if "group" in rnd:
            gs_fixtures.append(f)
        elif any(kw in rnd for kw in KNOCKOUT_KEYWORDS):
            ko_fixtures.append(f)
        else:
            print(f"    [IGNORADO] Rodada desconhecida: {f.get('league',{}).get('round','')}")

    print(f"\n  Fase de grupos: {len(gs_fixtures)} jogos")
    print(f"  Fase eliminatória (FT): {len(ko_fixtures)} jogos")
    return ko_fixtures


def build_ko_rows(ko_fixtures: list, existing_ids: set) -> list[dict]:
    """Converte fixtures KO em linhas para o raw CSV."""
    new_rows = []
    n_skip = 0

    for fixture in ko_fixtures:
        fid      = str(fixture["fixture"]["id"])
        rnd      = fixture.get("league", {}).get("round", "")
        home_raw = fixture["teams"]["home"]["name"]
        away_raw = fixture["teams"]["away"]["name"]
        home_n   = NAME_MAP.get(home_raw, home_raw)
        away_n   = NAME_MAP.get(away_raw, away_raw)
        date     = fixture["fixture"]["date"][:10]
        goals    = fixture["goals"]
        gh       = goals.get("home", 0) or 0
        ga       = goals.get("away", 0) or 0

        # Resultado extra-time / penalties
        score_ft  = fixture.get("score", {}).get("fulltime", {})
        score_et  = fixture.get("score", {}).get("extratime", {})
        score_pen = fixture.get("score", {}).get("penalty", {})
        went_et   = score_et.get("home") is not None
        went_pen  = score_pen.get("home") is not None

        if fid in existing_ids:
            n_skip += 1
            continue

        # Buscar estatísticas
        stats_raw  = api_get("fixtures/statistics", {"fixture": int(fid)}, cache_id=f"wc2026_ko_{fid}")
        team_stats: dict[str, dict] = {}
        for entry in stats_raw.get("response", []):
            tname = entry["team"]["name"]
            team_stats[tname] = {s["type"]: s["value"] for s in entry.get("statistics", [])}

        for side, opp_side, t_raw, o_raw, gf, gc in [
            ("home", "away", home_raw, away_raw, gh, ga),
            ("away", "home", away_raw, home_raw, ga, gh),
        ]:
            t_n = NAME_MAP.get(t_raw, t_raw)
            o_n = NAME_MAP.get(o_raw, o_raw)
            row: dict = {
                "match_id":        fid,
                "date":            date,
                "competition":     "FIFA World Cup 2026",
                "confederation":   COPA_TEAM_CONF.get(t_n, "UEFA"),
                "match_type":      "WC",
                "season":          2026,
                "home_away":       side,
                "team":            t_n,
                "opponent":        o_n,
                "gols_marcados":   gf,
                "gols_sofridos":   gc,
                "shots":           None,
                "shots_on_goal":   None,
                "blocked_shots":   None,
                "ball_possession": None,
                "fouls":           None,
                "yellow_cards":    None,
                "red_cards":       None,
                "corners":         None,
                "offsides":        None,
                "passes_accurate": None,
                "saves":           None,
                "ast":             None,
                "has_shot_data":   False,
                "weight":          1.0,
                "home_advantage":  0,
                "ko_round":        rnd,
                "went_et":         went_et,
                "went_penalties":  went_pen,
            }
            stats = team_stats.get(t_raw) or team_stats.get(t_n)
            if stats:
                row["has_shot_data"] = True
                for api_type, col in STAT_MAP.items():
                    row[col] = _parse_val(stats.get(api_type))
            new_rows.append(row)

    print(f"  KO novos: {len(new_rows)//2} jogos  |  {n_skip} já existiam")
    return new_rows


def main():
    print("=" * 72)
    print("  Copa 2026 — Busca resultados R32 via API")
    print("=" * 72)

    raw = pd.read_csv(RAW_CSV)
    existing_ids = set(raw["match_id"].astype(str))
    print(f"  Raw atual: {len(raw)} linhas | {len(existing_ids)} jogos únicos")

    # Verificar último jogo no raw
    raw["date"] = pd.to_datetime(raw["date"])
    last_wc = raw[raw["match_type"] == "WC"].sort_values("date")
    if not last_wc.empty:
        last = last_wc.iloc[-1]
        print(f"  Último WC no raw: {last['date'].date()} — {last['team']} vs {last['opponent']}")

    print("\n  Buscando jogos eliminatórios FT via API...")
    ko_fixtures = fetch_all_ko_games()

    if not ko_fixtures:
        print("\n  Nenhum jogo eliminatório finalizado na API ainda.")
        return

    print("\n  Detalhes dos jogos KO encontrados:")
    for f in ko_fixtures:
        rnd      = f.get("league", {}).get("round", "")
        home_raw = f["teams"]["home"]["name"]
        away_raw = f["teams"]["away"]["name"]
        home_n   = NAME_MAP.get(home_raw, home_raw)
        away_n   = NAME_MAP.get(away_raw, away_raw)
        gh       = f["goals"].get("home", "?")
        ga       = f["goals"].get("away", "?")
        date     = f["fixture"]["date"][:10]
        fid      = str(f["fixture"]["id"])
        in_raw   = "JÁ NO DB" if fid in existing_ids else "NOVO"
        print(f"  [{in_raw}] {date}  {rnd}  {home_n} {gh}–{ga} {away_n}")

    print("\n  Construindo linhas para o CSV...")
    new_rows = build_ko_rows(ko_fixtures, existing_ids)

    if not new_rows:
        print("  Nenhuma linha nova. Dataset já atualizado.")
        return

    # Verificar colunas — adicionar as novas se necessário
    new_df = pd.DataFrame(new_rows)
    for col in new_df.columns:
        if col not in raw.columns:
            raw[col] = None

    # Adicionar ao raw
    raw_updated = pd.concat([raw, new_df], ignore_index=True)
    raw_updated.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"\n  Raw atualizado: {len(raw)} → {len(raw_updated)} linhas")
    print(f"  Arquivo: {RAW_CSV}")

    # Confirmar último jogo
    raw_updated["date"] = pd.to_datetime(raw_updated["date"])
    last_added = raw_updated[raw_updated["match_type"] == "WC"].sort_values("date").iloc[-1]
    print(f"  Último WC agora: {last_added['date'].date()} — {last_added['team']} vs {last_added['opponent']}")


if __name__ == "__main__":
    main()
