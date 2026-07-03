"""
diagnostics.py — Diagnostico completo do dataset para o modelo Copa 2026.

Blocos:
  1. Shape e cobertura
  2. Features disponíveis (dtypes, nulos, stats, histogramas, correlacao)
  3. Análise do target (distribuicao de gols, dispersao, placares)
  4. Volume por time (jogos com features completas)
  5. Janela móvel efetiva (quantos jogos de histórico por avg5)

Saída: terminal + outputs/diagnostics.txt
"""

import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV  = Path("data/processed/model_dataset.csv")
OUTPUT_TXT = Path("outputs/diagnostics.txt")
OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

AVG5_COLS_HOME = [
    "home_gls_avg5", "home_ast_avg5", "home_sh_avg5", "home_sot_avg5",
    "home_fls_avg5", "home_tklw_avg5", "home_int_avg5",
]
AVG5_COLS_AWAY = [
    "away_gls_avg5", "away_ast_avg5", "away_sh_avg5", "away_sot_avg5",
    "away_fls_avg5", "away_tklw_avg5", "away_int_avg5",
]
# Features genéricas (sem prefixo home/away) para correlacao e histogramas
FEAT_NAMES = ["gls_avg5", "ast_avg5", "sh_avg5", "sot_avg5",
              "fls_avg5", "tklw_avg5", "int_avg5"]

# Para Block 5: quais features sao calculadas apenas com jogos CONMEBOL
CONMEBOL_ONLY_FEATS = {"sh_avg5", "sot_avg5", "fls_avg5", "tklw_avg5", "int_avg5"}
ALL_GAMES_FEATS     = {"gls_avg5", "ast_avg5"}


# ---------------------------------------------------------------------------
# Utilidades de output (Tee: terminal + arquivo)
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def sep(char="=", width=70):
    print(char * width)


def h1(title: str):
    sep()
    print(f"  {title}")
    sep()


def h2(title: str):
    sep("-", 70)
    print(f"  {title}")
    sep("-", 70)


# ---------------------------------------------------------------------------
# Bloco 1 — Shape e cobertura
# ---------------------------------------------------------------------------

def bloco1(df: pd.DataFrame):
    h1("BLOCO 1 -- SHAPE E COBERTURA")

    print(f"\nLinhas (jogos) : {len(df)}")
    print(f"Colunas        : {len(df.columns)}")

    # Competicao: home_stats_complete=True -> CONMEBOL, False -> friendly
    df["_comp"] = df["home_stats_complete"].map({True: "CONMEBOL", False: "friendly"})
    print("\nJogos por competicao:")
    for comp, cnt in df["_comp"].value_counts().items():
        print(f"  {comp:<12s}: {cnt}")

    # Por ano
    df["_year"] = df["date"].dt.year
    print("\nJogos por ano:")
    for yr, cnt in df["_year"].value_counts().sort_index().items():
        print(f"  {yr}: {cnt}")

    # Por mes (todos os anos juntos)
    df["_month"] = df["date"].dt.month
    print("\nJogos por mes (total):")
    for m, cnt in df["_month"].value_counts().sort_index().items():
        mname = ["Jan","Fev","Mar","Abr","Mai","Jun",
                 "Jul","Ago","Set","Out","Nov","Dez"][m - 1]
        print(f"  {mname}: {cnt}")

    # Times unicos
    home_teams = set(df["home_team"].unique())
    away_teams = set(df["away_team"].unique())
    all_teams  = home_teams | away_teams
    print(f"\nTimes unicos como home : {len(home_teams)}")
    print(f"Times unicos como away : {len(away_teams)}")
    print(f"Times unicos (total)   : {len(all_teams)}")
    print(f"  {sorted(all_teams)}")


# ---------------------------------------------------------------------------
# Bloco 2 — Features disponíveis
# ---------------------------------------------------------------------------

def text_histogram(series: pd.Series, bins: int = 10, width: int = 40) -> str:
    clean = series.dropna()
    if len(clean) == 0:
        return "  (sem dados)"
    counts, edges = np.histogram(clean, bins=bins)
    max_c = max(counts) if max(counts) > 0 else 1
    lines = []
    for i, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts)):
        bar = "#" * int(c / max_c * width)
        lines.append(f"  [{lo:5.2f},{hi:5.2f}) |{bar:<{width}}| {c}")
    return "\n".join(lines)


def bloco2(df: pd.DataFrame):
    h1("BLOCO 2 -- FEATURES DISPONIVEIS")

    # Stats por coluna
    h2("Dtypes, nulos e estatisticas descritivas")
    num_cols = df.select_dtypes(include="number").columns.tolist()

    header = f"{'Coluna':<22s} {'dtype':<10s} {'%null':>6s} {'min':>8s} {'max':>8s} {'media':>8s} {'std':>8s}"
    print(header)
    print("-" * len(header))
    for col in df.columns:
        dtype = str(df[col].dtype)
        pnull = df[col].isna().mean() * 100
        if col in num_cols:
            vmin  = df[col].min()
            vmax  = df[col].max()
            vmean = df[col].mean()
            vstd  = df[col].std()
            print(f"{col:<22s} {dtype:<10s} {pnull:6.1f}% {vmin:8.2f} {vmax:8.2f} {vmean:8.2f} {vstd:8.2f}")
        else:
            n_unique = df[col].nunique()
            print(f"{col:<22s} {dtype:<10s} {pnull:6.1f}%  (categorico, {n_unique} unicos)")

    # Histogramas das avg5
    h2("Histogramas das colunas avg5 (home + away combinados)")
    for feat in FEAT_NAMES:
        home_col = f"home_{feat}"
        away_col = f"away_{feat}"
        combined = pd.concat([df[home_col], df[away_col]], ignore_index=True)
        print(f"\n  {feat}  (n={combined.notna().sum()} obs, "
              f"med={combined.mean():.2f}, std={combined.std():.2f}):")
        print(text_histogram(combined))

    # Matriz de correlacao (avg5 home + away juntos, media)
    h2("Matriz de correlacao entre features avg5")
    feat_df = pd.DataFrame({
        f: pd.concat([df[f"home_{f}"], df[f"away_{f}"]], ignore_index=True)
        for f in FEAT_NAMES
    })
    corr = feat_df.corr()

    # Impressao formatada
    print(f"\n{'':>12s}", end="")
    for f in FEAT_NAMES:
        print(f"{f[:9]:>10s}", end="")
    print()
    for f1 in FEAT_NAMES:
        print(f"{f1[:12]:<12s}", end="")
        for f2 in FEAT_NAMES:
            val = corr.loc[f1, f2]
            print(f"{val:10.3f}", end="")
        print()

    # Pares com correlacao > 0.7
    h2("Pares com correlacao > 0.7 (candidatos a colinearidade)")
    found = False
    for i, f1 in enumerate(FEAT_NAMES):
        for f2 in FEAT_NAMES[i + 1:]:
            val = corr.loc[f1, f2]
            if abs(val) > 0.7:
                print(f"  {f1} x {f2}: {val:.3f}")
                found = True
    if not found:
        print("  Nenhum par com |corr| > 0.7")


# ---------------------------------------------------------------------------
# Bloco 3 — Analise do target
# ---------------------------------------------------------------------------

def bloco3(df: pd.DataFrame):
    h1("BLOCO 3 -- ANALISE DO TARGET (GOLS)")

    all_gols = pd.concat([df["home_gols"], df["away_gols"]], ignore_index=True)

    # Distribuicao discreta
    h2("Distribuicao de home_gols e away_gols")
    for side, col in [("home", "home_gols"), ("away", "away_gols")]:
        counts = df[col].value_counts().sort_index()
        print(f"\n  {side}_gols:")
        for g, c in counts.items():
            stars = "#" * c
            label = f"{g}+" if g >= 4 else str(g)
            if g == 4:
                cnt4plus = (df[col] >= 4).sum()
                print(f"    4+  gols: {cnt4plus:3d} {('#' * cnt4plus)}")
                break
            print(f"    {label}   gols: {c:3d}  {stars}")

    # Media e variancia — teste de adequacao Poisson
    h2("Media vs variancia (adequacao ao modelo de Poisson)")
    for side, col in [("home", "home_gols"), ("away", "away_gols")]:
        mu  = df[col].mean()
        var = df[col].var()
        ratio = var / mu
        adequado = "OK (Poisson adequado)" if ratio < 1.5 else "ATENCAO: variancia >> media — considerar Negative Binomial"
        print(f"  {col}: media={mu:.3f}  variancia={var:.3f}  var/media={ratio:.3f}  -> {adequado}")

    mu_all  = all_gols.mean()
    var_all = all_gols.var()
    print(f"\n  Combinado (home+away): media={mu_all:.3f}  var={var_all:.3f}  ratio={var_all/mu_all:.3f}")

    # Gols por competicao
    h2("Media de gols por competicao")
    conmebol = df[df["home_stats_complete"]]
    friendly = df[~df["home_stats_complete"]]
    for label, sub in [("CONMEBOL", conmebol), ("friendly", friendly)]:
        if len(sub) == 0:
            continue
        gh = sub["home_gols"].mean()
        ga = sub["away_gols"].mean()
        gt = (sub["home_gols"] + sub["away_gols"]).mean()
        print(f"  {label:<12s}: home={gh:.2f}  away={ga:.2f}  total_jogo={gt:.2f}  (n={len(sub)})")

    # Placar mais comum
    h2("Placares mais comuns")
    df["_placar"] = df["home_gols"].astype(str) + "-" + df["away_gols"].astype(str)
    top = df["_placar"].value_counts().head(10)
    for placar, cnt in top.items():
        pct = cnt / len(df) * 100
        print(f"  {placar:>6s}: {cnt:3d} vezes ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Bloco 4 — Volume por time
# ---------------------------------------------------------------------------

def bloco4(df: pd.DataFrame):
    h1("BLOCO 4 -- VOLUME POR TIME (JOGOS COM FEATURES COMPLETAS)")

    avg5_cols = [c for c in df.columns if c.endswith("_avg5")]

    # Para cada time: contar jogos onde esse time tem avg5 nao-nula
    team_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        for side, tcol in [("home", "home_team"), ("away", "away_team")]:
            team = row[tcol]
            feat_cols = [c for c in avg5_cols if c.startswith(side)]
            has_feats = not any(pd.isna(row[c]) for c in feat_cols)
            if has_feats:
                team_counts[team] = team_counts.get(team, 0) + 1

    sorted_teams = sorted(team_counts.items(), key=lambda x: -x[1])

    print(f"\n  {'Time':<20s} {'Jogos c/ features completas':>30s}")
    print("  " + "-" * 52)
    for team, cnt in sorted_teams:
        flag = " << POUCOS DADOS" if cnt < 5 else ""
        print(f"  {team:<20s} {cnt:>30d}{flag}")

    # Times com menos de 5 jogos
    problematic = [(t, c) for t, c in sorted_teams if c < 5]
    print(f"\n  Times com < 5 jogos completos: {len(problematic)}")
    for t, c in problematic:
        print(f"    {t}: {c} jogos")

    # Times sem nenhum jogo completo
    all_teams = set(df["home_team"]) | set(df["away_team"])
    zero_teams = sorted(all_teams - set(team_counts.keys()))
    if zero_teams:
        print(f"\n  Times SEM nenhum jogo com features completas ({len(zero_teams)}):")
        for t in zero_teams:
            print(f"    {t}")
    else:
        print("\n  Todos os times tem ao menos 1 jogo com features completas.")


# ---------------------------------------------------------------------------
# Bloco 5 — Janela movel efetiva
# ---------------------------------------------------------------------------

def bloco5(df: pd.DataFrame):
    h1("BLOCO 5 -- JANELA MOVEL EFETIVA (quantos jogos de historico por avg5)")

    df = df.copy().sort_values("date").reset_index(drop=True)

    # Reconstroe contagem de jogos anteriores por time e tipo de feature
    # all_games_count[team]: total de jogos vistos para esse time ate agora (para gls/ast)
    # conmebol_count[team]: total de jogos CONMEBOL vistos para esse time (para sh/sot/etc)
    all_games_count: dict[str, int] = {}
    conmebol_count:  dict[str, int] = {}

    # Para cada coluna avg5, guarda lista de "janela efetiva" (min(hist,5) ou 0 se NaN)
    # Processamos em ordem cronologica para obter janela real
    window_sizes: dict[str, list[int]] = {f: [] for f in FEAT_NAMES}

    for _, row in df.iterrows():
        is_conmebol = bool(row["home_stats_complete"])

        for side, tcol in [("home", "home_team"), ("away", "away_team")]:
            team = row[tcol]

            all_g = all_games_count.get(team, 0)
            con_g = conmebol_count.get(team, 0)

            for feat in FEAT_NAMES:
                col = f"{side}_{feat}"
                val = row[col]
                if pd.isna(val):
                    effective = 0
                elif feat in CONMEBOL_ONLY_FEATS:
                    effective = min(con_g, 5)
                else:
                    effective = min(all_g, 5)
                window_sizes[feat].append(effective)

        # Atualiza contadores APOS processar o jogo (anti-leakage)
        for team in [row["home_team"], row["away_team"]]:
            all_games_count[team] = all_games_count.get(team, 0) + 1
            if is_conmebol:
                conmebol_count[team] = conmebol_count.get(team, 0) + 1

    # Imprime distribuicao por feature
    h2("Distribuicao da janela efetiva por feature (0=NaN, 1-5=historico usado)")
    for feat in FEAT_NAMES:
        sizes = window_sizes[feat]
        total = len(sizes)
        counts = {k: 0 for k in range(6)}
        for s in sizes:
            counts[s] += 1

        note = "(CONMEBOL apenas)" if feat in CONMEBOL_ONLY_FEATS else "(CONMEBOL + friendly)"
        print(f"\n  {feat}  {note}:")
        print(f"  {'Janela':<8s} {'N':>6s} {'%':>7s}  {'Barra'}")
        print("  " + "-" * 50)
        for k in range(6):
            c   = counts[k]
            pct = c / total * 100
            bar = "#" * int(pct / 2)
            lbl = "NaN/0" if k == 0 else f"    {k}"
            print(f"  {lbl:<8s} {c:6d} {pct:7.1f}%  {bar}")

        # Resumo: entre os nao-nulos, qual a janela media
        non_zero = [s for s in sizes if s > 0]
        if non_zero:
            avg_w = np.mean(non_zero)
            print(f"  => Janela media (excl. NaN): {avg_w:.2f}  "
                  f"(de {len(non_zero)} obs nao-nulas)")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    buf = StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, buf)

    try:
        df = pd.read_csv(INPUT_CSV, parse_dates=["date"])

        sep()
        print("  DIAGNOSTICO COMPLETO -- COPA 2026 DATASET")
        print(f"  Arquivo: {INPUT_CSV}")
        sep()
        print()

        bloco1(df)
        print()
        bloco2(df)
        print()
        bloco3(df)
        print()
        bloco4(df)
        print()
        bloco5(df)

        sep()
        print("  FIM DO DIAGNOSTICO")
        sep()

    finally:
        sys.stdout = original_stdout

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"\nOutput completo salvo em: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
