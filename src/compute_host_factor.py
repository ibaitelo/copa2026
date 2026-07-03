"""
compute_host_factor.py — Calcula fator de vantagem de sede para Copa 2026.

Etapa 1: lê intl_results.csv (resultados históricos de Copas do Mundo 1998-2022)
         identifica partidas dos times-sede jogando no próprio país e calcula
         o excesso de gols e win_rate em relação à média geral da Copa.

Etapa 2: projeta o host_factor para Copa 2026 (USA, México, Canadá) com
         ajuste pelo ELO relativo de cada time-sede em relação aos 48 da Copa.

Saídas:
  data/external/host_advantage_historical.csv  — estatísticas por Copa/sede
  data/external/host_factor_copa2026.csv        — host_factor por time para 2026
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INT_RESULTS = Path("data/raw/intl_results.csv")
ELO_CSV     = Path("data/raw/elo_history.csv")
OUT_HIST    = Path("data/external/host_advantage_historical.csv")
OUT_2026    = Path("data/external/host_factor_copa2026.csv")

# --------------------------------------------------------------------------
# Hosts históricos de Copa (jogos disputados no país sede)
# --------------------------------------------------------------------------
WC_HOSTS: dict[int, list[str]] = {
    1998: ["France"],
    2002: ["South Korea", "Japan"],
    2006: ["Germany"],
    2010: ["South Africa"],
    2014: ["Brazil"],
    2018: ["Russia"],
    2022: ["Qatar"],
}

# Mapeamento team → country (campo "country" em intl_results.csv)
TEAM_TO_COUNTRY: dict[str, str] = {
    "France":       "France",
    "South Korea":  "South Korea",
    "Japan":        "Japan",
    "Germany":      "Germany",
    "South Africa": "South Africa",
    "Brazil":       "Brazil",
    "Russia":       "Russia",
    "Qatar":        "Qatar",
    # Copa 2026
    "United States": "United States",
    "Mexico":        "Mexico",
    "Canada":        "Canada",
}

# Times Copa 2026 (os 48 classificados)
COPA2026_TEAMS = [
    "Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay", "Paraguay",
    "Germany", "France", "Spain", "England", "Netherlands", "Portugal",
    "Belgium", "Austria", "Switzerland", "Croatia",
    "Norway", "Sweden", "Czech Republic", "Turkey",
    "Scotland", "Bosnia and Herzegovina",
    "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia",
    "Uzbekistan", "Jordan", "Iraq", "Qatar",
    "Morocco", "Senegal", "Egypt", "Ghana", "Cape Verde",
    "DR Congo", "Ivory Coast", "South Africa", "Algeria", "Tunisia",
    "United States", "Mexico", "Canada", "Panama", "Curacao", "Haiti",
    "New Zealand",
]

HOST_2026 = ["United States", "Mexico", "Canada"]


# --------------------------------------------------------------------------
# ETAPA 1 — Calcular host advantage histórico
# --------------------------------------------------------------------------

def load_wc_games() -> pd.DataFrame:
    df = pd.read_csv(INT_RESULTS)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    wc = df[df["tournament"] == "FIFA World Cup"].copy()
    return wc


def build_long_wc(wc: pd.DataFrame) -> pd.DataFrame:
    """Converte wide (home/away por jogo) para long (1 linha por time por jogo)."""
    rows = []
    for _, r in wc.iterrows():
        for side, opp_side, gf, ga in [
            ("home", "away", r["home_score"], r["away_score"]),
            ("away", "home", r["away_score"], r["home_score"]),
        ]:
            team = r[f"{side}_team"]
            rows.append({
                "year":    r["year"],
                "date":    r["date"],
                "team":    team,
                "opponent": r[f"{opp_side}_team"],
                "gf":      float(gf) if pd.notna(gf) else np.nan,
                "ga":      float(ga) if pd.notna(ga) else np.nan,
                "country": r["country"],
                "neutral": r["neutral"],
            })
    return pd.DataFrame(rows)


def compute_host_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada Copa e cada time-sede, calcula:
      - avg_gf_home:  média de gols marcados jogando no próprio país
      - avg_gf_away:  média de gols marcados fora do próprio país (na mesma Copa)
      - win_rate_home / win_rate_away
    """
    records = []
    for year, hosts in WC_HOSTS.items():
        year_df = long_df[long_df["year"] == year].copy()
        if year_df.empty:
            continue

        # Média geral de gols por jogo nessa Copa (todos os times)
        wc_avg_gf = year_df["gf"].mean()

        for team in hosts:
            host_country = TEAM_TO_COUNTRY.get(team)
            team_df = year_df[year_df["team"] == team].copy()
            if team_df.empty:
                continue

            team_df["is_home_country"] = team_df["country"] == host_country

            home_g = team_df[team_df["is_home_country"]]
            away_g = team_df[~team_df["is_home_country"]]

            def _metrics(sub: pd.DataFrame) -> dict:
                if sub.empty:
                    return {"avg_gf": np.nan, "win_rate": np.nan, "n_games": 0}
                valid = sub.dropna(subset=["gf", "ga"])
                avg_gf    = valid["gf"].mean() if not valid.empty else np.nan
                win_rate  = (valid["gf"] > valid["ga"]).mean() if not valid.empty else np.nan
                return {"avg_gf": avg_gf, "win_rate": win_rate, "n_games": len(valid)}

            h = _metrics(home_g)
            a = _metrics(away_g)

            # Multiplicador de gols: quanto a mais o time-sede marca jogando em casa
            # vs a média geral da Copa naquele ano
            boost_vs_avg = (h["avg_gf"] / wc_avg_gf - 1.0) if (pd.notna(h["avg_gf"]) and wc_avg_gf > 0) else np.nan
            boost_vs_own_away = (h["avg_gf"] / a["avg_gf"] - 1.0) if (pd.notna(h["avg_gf"]) and pd.notna(a["avg_gf"]) and a["avg_gf"] > 0) else np.nan

            records.append({
                "year":           year,
                "host_team":      team,
                "wc_avg_gf":      round(wc_avg_gf, 4),
                "home_avg_gf":    round(h["avg_gf"], 4)    if pd.notna(h["avg_gf"]) else np.nan,
                "home_win_rate":  round(h["win_rate"], 4)   if pd.notna(h["win_rate"]) else np.nan,
                "home_n_games":   h["n_games"],
                "away_avg_gf":    round(a["avg_gf"], 4)    if pd.notna(a["avg_gf"]) else np.nan,
                "away_win_rate":  round(a["win_rate"], 4)   if pd.notna(a["win_rate"]) else np.nan,
                "away_n_games":   a["n_games"],
                "boost_vs_wc_avg":   round(boost_vs_avg, 4)     if pd.notna(boost_vs_avg) else np.nan,
                "boost_vs_own_away": round(boost_vs_own_away, 4) if pd.notna(boost_vs_own_away) else np.nan,
            })

    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# ETAPA 2 — Projetar host_factor para Copa 2026
# --------------------------------------------------------------------------

def load_current_elo() -> dict[str, float]:
    if not ELO_CSV.exists():
        return {}
    elo = pd.read_csv(ELO_CSV, parse_dates=["date"])
    return (elo.sort_values("date").groupby("team")["elo_after"].last().to_dict())


def compute_copa2026_host_factors(hist: pd.DataFrame,
                                   elo_map: dict) -> pd.DataFrame:
    """
    host_factor = host_factor_base × elo_scaling

    host_factor_base: média do boost_vs_wc_avg entre os 9 casos históricos.
    elo_scaling     : min(1.0, max(0.4, team_elo / copa2026_mean_elo))
                      times muito abaixo da média recebem fator reduzido.

    Para jogos NO próprio país  : host_factor pleno.
    Para jogos EM CO-HOST       : 0.5 × host_factor (caso México nos EUA, etc.).
    """
    valid = hist["boost_vs_wc_avg"].dropna()
    base  = float(valid.mean()) if not valid.empty else 0.15
    base  = max(0.05, min(0.40, base))   # clamp razoável

    # ELO médio dos 48 times Copa 2026
    copa_elos = [elo_map[t] for t in COPA2026_TEAMS if t in elo_map]
    elo_mean  = np.mean(copa_elos) if copa_elos else 1500.0

    rows = []
    for team in HOST_2026:
        elo_val = elo_map.get(team, elo_mean)
        elo_scale = min(1.0, max(0.4, elo_val / elo_mean))
        hf_full    = round(base * elo_scale, 4)
        hf_cohost  = round(hf_full * 0.5, 4)
        rows.append({
            "team":              team,
            "elo":               round(elo_val),
            "elo_mean_copa2026": round(elo_mean),
            "elo_scale":         round(elo_scale, 4),
            "host_factor_base":  round(base, 4),
            "host_factor_home":  hf_full,    # jogando no próprio país
            "host_factor_cohost": hf_cohost, # jogando em co-host
        })

    # Todos os outros times: host_factor = 0
    for team in COPA2026_TEAMS:
        if team not in HOST_2026:
            rows.append({
                "team": team, "elo": round(elo_map.get(team, 0)),
                "elo_mean_copa2026": round(elo_mean),
                "elo_scale": 0, "host_factor_base": round(base, 4),
                "host_factor_home": 0.0, "host_factor_cohost": 0.0,
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  compute_host_factor.py — Vantagem de Sede (Copa 2026)")
    print(sep)

    print("\n  [1] Carregando histórico de Copas (1998-2022)...")
    wc  = load_wc_games()
    lng = build_long_wc(wc)
    print(f"      {len(wc)} jogos |  {lng['year'].nunique()} Copas")

    hist = compute_host_stats(lng)

    print(f"\n  [2] Estatísticas por time-sede:")
    print(f"\n  {'Copa':>5}  {'Sede':15}  {'GF@Home':>8}  {'GF@Away':>8}  {'Boost%':>7}  "
          f"{'WR@Home':>8}  {'WR@Away':>8}")
    print("  " + "-" * 67)
    for _, r in hist.iterrows():
        boost_str = f"{r['boost_vs_wc_avg']*100:+.1f}%" if pd.notna(r['boost_vs_wc_avg']) else "   N/A"
        gf_h = f"{r['home_avg_gf']:.2f}" if pd.notna(r['home_avg_gf']) else " NaN"
        gf_a = f"{r['away_avg_gf']:.2f}" if pd.notna(r['away_avg_gf']) else " NaN"
        wr_h = f"{r['home_win_rate']*100:.0f}%" if pd.notna(r['home_win_rate']) else " NaN"
        wr_a = f"{r['away_win_rate']*100:.0f}%" if pd.notna(r['away_win_rate']) else " NaN"
        print(f"  {int(r['year']):5d}  {r['host_team']:15}  {gf_h:>8}  {gf_a:>8}  "
              f"{boost_str:>7}  {wr_h:>8}  {wr_a:>8}")

    valid = hist["boost_vs_wc_avg"].dropna()
    base  = float(valid.mean())
    print(f"\n  host_factor_base (média histórica) = {base:+.4f} ({base*100:+.1f}%)")
    print(f"  Intervalo: [{valid.min()*100:+.1f}% , {valid.max()*100:+.1f}%]")

    OUT_HIST.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(OUT_HIST, index=False, encoding="utf-8")
    print(f"\n  Salvo: {OUT_HIST}")

    print(f"\n  [3] Projetando host_factor para Copa 2026...")
    elo_map  = load_current_elo()
    hf_df    = compute_copa2026_host_factors(hist, elo_map)

    hosts_df = hf_df[hf_df["host_factor_home"] > 0]
    print(f"\n  {'Time':15}  {'ELO':>6}  {'ELO_scale':>10}  "
          f"{'HF_home':>8}  {'HF_cohost':>10}")
    print("  " + "-" * 54)
    for _, r in hosts_df.iterrows():
        print(f"  {r['team']:15}  {int(r['elo']):6d}  {r['elo_scale']:>10.4f}  "
              f"{r['host_factor_home']:>8.4f}  {r['host_factor_cohost']:>10.4f}")

    OUT_2026.parent.mkdir(parents=True, exist_ok=True)
    hf_df.to_csv(OUT_2026, index=False, encoding="utf-8")
    print(f"\n  Salvo: {OUT_2026}")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
