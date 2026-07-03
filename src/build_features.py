"""
build_features.py — Constrói dataset para o modelo de predição de placares.

Fonte : data/raw/full_dataset_raw.csv  (1 linha por time por jogo)
Saída : data/processed/model_dataset.csv

Features com decay temporal exponencial (substitui janela fixa avg5):
  w_final(t) = exp(-ξ × dias_antes) × match_weight
  match_weight: WCQ/FIFA=1.0, amistosos=0.7
  dias_antes = dias entre jogo histórico e jogo a prever
  Anti-leakage: apenas jogos estritamente anteriores à data D.

  gls_decay, ast_decay, shots_on_goal_decay, blocked_shots_decay,
  ball_possession_decay, fouls_decay, corners_decay, saves_decay,
  win_rate_decay, has_decay_data

  gls_decay_vs_forte / gls_decay_vs_fraco  (opp_rating > 6.8 / ≤ 6.8, min_periods=2)
  shots_decay_vs_forte, gls_ponderado_decay,
  delta_rating_decay, elo_diff_decay, rating_titulares_decay
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RATINGS_CSV         = Path("data/raw/player_ratings_per_game.csv")
STARTER_RATINGS_CSV = Path("data/raw/starter_ratings_per_game.csv")
ELO_HISTORY_CSV     = Path("data/raw/elo_history.csv")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INPUT_CSV  = Path("data/raw/full_dataset_raw.csv")
OUTPUT_CSV = Path("data/processed/model_dataset.csv")

# ξ padrão — otimizado via CV em model_xgboost.py
XI_DEFAULT = 0.003

MATCH_WEIGHT_WCQ      = 1.0
MATCH_WEIGHT_FRIENDLY = 0.7
HAS_DECAY_MIN_GAMES   = 3  # mínimo de jogos para has_decay_data = True

# v9 — calibração ELO para gls_ponderado_decay e gls_decay_vs_forte/fraco
ELO_MEDIO_GLOBAL    = 1700.0  # média ELO global (todas as seleções FIFA ativas)
ELO_FORTE_THRESHOLD = 1750    # ELO mínimo para adversário ser "forte" (mediana Copa 2026)

# min_obs por coluna fonte (mínimo de valores não-NaN para calcular o decay)
MIN_OBS: dict[str, int] = {
    "shots_on_goal": 3,
    "blocked_shots": 3,
}
_DEFAULT_MIN_OBS = 1

SIMPLE_COLS: dict[str, str] = {
    "gols_marcados":   "gls_decay",
    "gols_sofridos":   "gols_sofridos_decay",   # ETAPA 3: déficit defensivo decay
    "ast":             "ast_decay",
    "shots_on_goal":   "shots_on_goal_decay",
    "blocked_shots":   "blocked_shots_decay",
    "ball_possession": "ball_possession_decay",
    "fouls":           "fouls_decay",
    "corners":         "corners_decay",
    "saves":           "saves_decay",
}

# ETAPA 4 — features removidas do modelo final (mas ainda computadas para uso interno):
#   gls_decay           → substituída por gls_ponderado_decay (qualidade-ponderada)
#   win_rate_decay      → colinear com margem_gols_decay + gls_ponderado_decay
#   rating_titulares_decay → muitos NaN, sinal fraco vs elo_diff_decay
RATING_COLS: list[str] = [
    "delta_rating_decay",
    "gls_decay_vs_forte",
    "gls_decay_vs_fraco",
    "shots_decay_vs_forte",
    "gls_ponderado_decay",
    "elo_diff_decay",
    "rating_titulares_decay",   # mantida no dataset mas não em FEATURES (v8)
    "margem_gols_decay",        # ETAPA 3: gls_ponderado_decay − gols_sofridos_decay
]

ALL_DECAY = list(SIMPLE_COLS.values()) + ["win_rate_decay"] + RATING_COLS

COPA2026_TEAMS = {
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
}


# ---------------------------------------------------------------------------
# PASSO 1 — Carregar e mesclar dados brutos
# ---------------------------------------------------------------------------

def _merge_ratings(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["rating_medio_game", "opp_rating_medio_game"]:
        df[col] = np.nan

    if not RATINGS_CSV.exists():
        print("  [ratings] player_ratings_per_game.csv não encontrado")
        return df

    ratings = pd.read_csv(RATINGS_CSV)
    ratings["match_id"] = ratings["match_id"].astype(str)
    df["match_id"]      = df["match_id"].astype(str)

    own = ratings[["match_id", "team", "rating_medio"]].rename(
        columns={"rating_medio": "rating_medio_game"})
    df = df.merge(own, on=["match_id", "team"], how="left", suffixes=("", "_r"))
    if "rating_medio_game_r" in df.columns:
        df["rating_medio_game"] = df["rating_medio_game_r"].combine_first(df["rating_medio_game"])
        df.drop(columns=["rating_medio_game_r"], inplace=True)

    opp = ratings[["match_id", "team", "rating_medio"]].rename(
        columns={"team": "opponent", "rating_medio": "opp_rating_medio_game"})
    df = df.merge(opp, on=["match_id", "opponent"], how="left", suffixes=("", "_r"))
    if "opp_rating_medio_game_r" in df.columns:
        df["opp_rating_medio_game"] = df["opp_rating_medio_game_r"].combine_first(
            df["opp_rating_medio_game"])
        df.drop(columns=["opp_rating_medio_game_r"], inplace=True)

    n_own = df["rating_medio_game"].notna().sum()
    n_opp = df["opp_rating_medio_game"].notna().sum()
    print(f"  [ratings] Merge OK — {n_own} linhas com rating próprio, {n_opp} com rating oponente")
    return df


def _merge_elo(df: pd.DataFrame) -> pd.DataFrame:
    df["elo_diff_game"] = np.nan
    df["opp_elo_game"]  = np.nan   # ELO absoluto do adversário (usado em gls_ponderado_decay v9)

    if not ELO_HISTORY_CSV.exists():
        print("  [ELO] elo_history.csv não encontrado")
        return df

    elo_before = pd.read_csv(ELO_HISTORY_CSV, parse_dates=["date"])
    elo_before = elo_before[["date", "team", "elo_before"]].sort_values(["team", "date"])

    df_sorted = df[["match_id", "date", "team", "opponent"]].copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])

    teams_in_data    = df_sorted["team"].unique()
    opp_in_data      = df_sorted["opponent"].unique()
    all_teams_needed = set(teams_in_data) | set(opp_in_data)
    elo_lookup = (elo_before[elo_before["team"].isin(all_teams_needed)]
                  .sort_values(["team", "date"]))

    results: list[dict] = []
    for team, grp in df_sorted.groupby("team"):
        grp_s  = grp.sort_values("date")
        elo_t  = elo_lookup[elo_lookup["team"] == team].sort_values("date")

        merged_t = pd.merge_asof(
            grp_s[["match_id", "date", "opponent"]],
            elo_t[["date", "elo_before"]].rename(columns={"elo_before": "elo_team"}),
            on="date", direction="backward",
        )
        opp_elo_map: dict[str, pd.DataFrame] = {
            opp: elo_lookup[elo_lookup["team"] == opp].sort_values("date")
            for opp in grp_s["opponent"].unique()
        }
        for _, row in merged_t.iterrows():
            opp         = row["opponent"]
            opp_elo_df  = opp_elo_map.get(opp, pd.DataFrame())
            elo_opp     = np.nan
            if not opp_elo_df.empty:
                mask = opp_elo_df["date"] < row["date"]
                if mask.any():
                    elo_opp = float(opp_elo_df[mask].iloc[-1]["elo_before"])
            elo_team = row.get("elo_team", np.nan)
            if pd.notna(elo_team) and pd.notna(elo_opp):
                results.append({"match_id": row["match_id"], "team": team,
                                 "elo_diff_game": float(elo_team) - float(elo_opp),
                                 "opp_elo_game":  float(elo_opp)})

    if results:
        elo_df = pd.DataFrame(results)
        df = df.merge(elo_df, on=["match_id", "team"], how="left", suffixes=("", "_r"))
        if "elo_diff_game_r" in df.columns:
            df["elo_diff_game"] = df["elo_diff_game_r"].combine_first(df["elo_diff_game"])
            df.drop(columns=["elo_diff_game_r"], inplace=True)
        if "opp_elo_game_r" in df.columns:
            df["opp_elo_game"] = df["opp_elo_game_r"].combine_first(df["opp_elo_game"])
            df.drop(columns=["opp_elo_game_r"], inplace=True)

    n = df["elo_diff_game"].notna().sum()
    print(f"  [ELO] elo_diff_game: {n} linhas ({n/len(df)*100:.1f}%)")
    return df


def _merge_starter_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df["rating_titulares_game"] = np.nan

    if not STARTER_RATINGS_CSV.exists():
        print("  [Starters] starter_ratings_per_game.csv não encontrado")
        return df

    starters = pd.read_csv(STARTER_RATINGS_CSV)
    starters["match_id"] = starters["match_id"].astype(str)
    df["match_id"]       = df["match_id"].astype(str)

    starters = starters[["match_id", "team", "rating_titulares"]].rename(
        columns={"rating_titulares": "rating_titulares_game"})
    df = df.merge(starters, on=["match_id", "team"], how="left", suffixes=("", "_r"))
    if "rating_titulares_game_r" in df.columns:
        df["rating_titulares_game"] = df["rating_titulares_game_r"].combine_first(
            df["rating_titulares_game"])
        df.drop(columns=["rating_titulares_game_r"], inplace=True)

    n = df["rating_titulares_game"].notna().sum()
    print(f"  [Starters] rating_titulares_game: {n} linhas ({n/len(df)*100:.1f}%)")
    return df


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    num_cols = (["gols_marcados", "gols_sofridos", "weight"]
                + list(SIMPLE_COLS.keys()) + ["ast"])
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["has_shot_data"] = df["has_shot_data"].map(
        {"True": True, "False": False, True: True, False: False}
    ).fillna(False)

    print("PASSO 1 — Dados carregados")
    print(f"  Total linhas: {len(df):,}  |  jogos únicos: {df['match_id'].nunique()}")
    for mt, cnt in df.groupby("match_type")["match_id"].nunique().items():
        print(f"    {mt:<10}: {cnt} jogos")
    return df


def load_and_merge_raw() -> pd.DataFrame:
    """Carrega dados brutos e mescla ratings/ELO/starters (usado pelo tune_xi)."""
    df = load_data()
    df = _merge_ratings(df)
    df = _merge_elo(df)
    df = _merge_starter_ratings(df)
    return df


# ---------------------------------------------------------------------------
# PASSO 2 — Decay temporal exponencial por time
# ---------------------------------------------------------------------------

def add_decay_features(df: pd.DataFrame, xi: float = XI_DEFAULT,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Calcula features com decay exponencial: w(t) = exp(-ξ × dias_antes) × match_weight.
    Anti-leakage garantido: apenas jogos estritamente anteriores à data D.
    """
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    for col in ALL_DECAY:
        df[col] = np.nan
    df["has_decay_data"] = False

    # match_weight global (indexado pelo df.index)
    mw_global = np.where(df["match_type"].isin(["WCQ", "WC"]),
                         MATCH_WEIGHT_WCQ, MATCH_WEIGHT_FRIENDLY).astype(float)

    for team, grp_orig in df.groupby("team", sort=False):
        grp      = grp_orig.sort_values("date")
        orig_idx = grp.index.values   # índices no df original
        n        = len(grp)
        if n == 0:
            continue

        dates_np = grp["date"].values                 # numpy datetime64[ns]
        mw_arr   = mw_global[orig_idx]

        def _col(name):
            return grp[name].values.astype(float) if name in grp.columns \
                   else np.full(n, np.nan)

        gls_arr   = _col("gols_marcados")
        gls_s_arr = _col("gols_sofridos")
        ast_arr   = _col("ast")
        sot_arr   = _col("shots_on_goal")
        blk_arr   = _col("blocked_shots")
        poss_arr  = _col("ball_possession")
        fouls_arr = _col("fouls")
        corn_arr  = _col("corners")
        sv_arr    = _col("saves")
        own_r_arr = _col("rating_medio_game")
        opp_r_arr = _col("opp_rating_medio_game")
        elo_arr     = _col("elo_diff_game")
        opp_elo_arr = _col("opp_elo_game")      # v9: ELO absoluto do adversário
        tit_arr     = _col("rating_titulares_game")

        # output buffers (0-indexed within this team)
        out      = {col: np.full(n, np.nan) for col in ALL_DECAY}
        has_data = np.zeros(n, dtype=bool)

        for i in range(n):
            ref_date = dates_np[i]
            past     = dates_np < ref_date   # strictly past (anti-leakage)

            if not past.any():
                continue

            past_days = ((ref_date - dates_np[past]) / np.timedelta64(1, "D")).astype(float)
            past_mw   = mw_arr[past]
            base_w    = np.exp(-xi * past_days) * past_mw

            # ── has_decay_data ──────────────────────────────────────────
            n_gls_valid = int((~np.isnan(gls_arr[past])).sum())
            has_data[i] = (n_gls_valid >= HAS_DECAY_MIN_GAMES)

            # ── helper: decay-weighted mean ─────────────────────────────
            def _dmean(arr, min_obs=_DEFAULT_MIN_OBS):
                v = arr[past]
                ok = ~np.isnan(v)
                if ok.sum() < min_obs:
                    return np.nan
                w = base_w[ok]
                s = w.sum()
                return float(np.dot(w, v[ok]) / s) if s > 0 else np.nan

            out["gls_decay"][i]             = _dmean(gls_arr)
            out["gols_sofridos_decay"][i]   = _dmean(gls_s_arr)
            out["ast_decay"][i]             = _dmean(ast_arr)
            out["shots_on_goal_decay"][i]   = _dmean(sot_arr, MIN_OBS.get("shots_on_goal", _DEFAULT_MIN_OBS))
            out["blocked_shots_decay"][i]   = _dmean(blk_arr, MIN_OBS.get("blocked_shots", _DEFAULT_MIN_OBS))
            out["ball_possession_decay"][i] = _dmean(poss_arr)
            out["fouls_decay"][i]           = _dmean(fouls_arr)
            out["corners_decay"][i]         = _dmean(corn_arr)
            out["saves_decay"][i]           = _dmean(sv_arr)

            # ── win_rate_decay ──────────────────────────────────────────
            pg = gls_arr[past]; ps = gls_s_arr[past]
            ok_w = ~(np.isnan(pg) | np.isnan(ps))
            if ok_w.sum() >= 1:
                pts  = np.where(pg[ok_w] > ps[ok_w], 1.0,
                                np.where(pg[ok_w] == ps[ok_w], 0.5, 0.0))
                w_wr = base_w[ok_w]
                s    = w_wr.sum()
                if s > 0:
                    out["win_rate_decay"][i] = float(np.dot(w_wr, pts) / s)

            # ── Rating features ─────────────────────────────────────────
            p_own = own_r_arr[past]; p_opp = opp_r_arr[past]

            # delta_rating_decay
            delta = p_own - p_opp
            ok_d  = ~np.isnan(delta)
            if ok_d.sum() >= 1:
                w_d = base_w[ok_d]; s = w_d.sum()
                if s > 0:
                    out["delta_rating_decay"][i] = float(np.dot(w_d, delta[ok_d]) / s)

            # BUG 2 FIX: threshold relativo ao ELO (1750) — universal e calibrado
            # Antes: p_opp > 6.8 (rating jogadores) variava por confederação sem sentido
            pg2        = gls_arr[past]
            opp_elo_p  = opp_elo_arr[past]
            ok_opp_elo = ~np.isnan(opp_elo_p)
            forte_m    = ok_opp_elo & (opp_elo_p > ELO_FORTE_THRESHOLD) & (~np.isnan(pg2))
            fraco_m    = ok_opp_elo & (opp_elo_p <= ELO_FORTE_THRESHOLD) & (~np.isnan(pg2))

            if forte_m.sum() >= 2:
                wf = base_w[forte_m]; s = wf.sum()
                if s > 0:
                    out["gls_decay_vs_forte"][i] = float(np.dot(wf, pg2[forte_m]) / s)
            if fraco_m.sum() >= 2:
                wfr = base_w[fraco_m]; s = wfr.sum()
                if s > 0:
                    out["gls_decay_vs_fraco"][i] = float(np.dot(wfr, pg2[fraco_m]) / s)

            # shots_decay_vs_forte: mesmo threshold ELO (consistência)
            ps2    = sot_arr[past]
            fsot_m = ok_opp_elo & (opp_elo_p > ELO_FORTE_THRESHOLD) & (~np.isnan(ps2))
            if fsot_m.sum() >= 2:
                wfs = base_w[fsot_m]; s = wfs.sum()
                if s > 0:
                    out["shots_decay_vs_forte"][i] = float(np.dot(wfs, ps2[fsot_m]) / s)

            # BUG 1 FIX: gls_ponderado_decay com ponderação ELO absoluto
            # Fórmula correta: mean(gols × elo_opp/elo_medio, weights=decay_w)
            # Gol contra time forte (ELO 1900) vale 1900/1700=1.12 gols; fraco (ELO 1500) vale 0.88
            # Antes: cw = decay_w × opp_rating (ponderação invertida — fortes penalizavam)
            ok_pr = ok_opp_elo & (~np.isnan(pg2))
            if ok_pr.sum() >= 2:
                quality_adj = pg2[ok_pr] * (opp_elo_p[ok_pr] / ELO_MEDIO_GLOBAL)
                cw = base_w[ok_pr]; s = cw.sum()
                if s > 0:
                    out["gls_ponderado_decay"][i] = float(np.dot(cw, quality_adj) / s)

            # margem_gols_decay: ataque qualidade-ponderado − defesa decay
            # Captura saldo de gols ajustado pela qualidade dos adversários
            gp = out["gls_ponderado_decay"][i]
            gs = out["gols_sofridos_decay"][i]
            if pd.notna(gp) and pd.notna(gs):
                out["margem_gols_decay"][i] = gp - gs

            # elo_diff_decay
            pe = elo_arr[past]; ok_e = ~np.isnan(pe)
            if ok_e.sum() >= 1:
                we = base_w[ok_e]; s = we.sum()
                if s > 0:
                    out["elo_diff_decay"][i] = float(np.dot(we, pe[ok_e]) / s)

            # rating_titulares_decay
            pt = tit_arr[past]; ok_t = ~np.isnan(pt)
            if ok_t.sum() >= 1:
                wt = base_w[ok_t]; s = wt.sum()
                if s > 0:
                    out["rating_titulares_decay"][i] = float(np.dot(wt, pt[ok_t]) / s)

            # Structural zeros when has_decay_data=False indicate missing data,
            # not genuine values. Reset to NaN so confederation-median imputation applies.
            if not has_data[i]:
                if np.isnan(out["saves_decay"][i]) or out["saves_decay"][i] == 0.0:
                    out["saves_decay"][i] = np.nan
                if np.isnan(out["rating_titulares_decay"][i]) or out["rating_titulares_decay"][i] == 0.0:
                    out["rating_titulares_decay"][i] = np.nan

        # bulk assignment
        for col, arr in out.items():
            df.loc[orig_idx, col] = arr
        df.loc[orig_idx, "has_decay_data"] = has_data

    # ── Anti-leakage assert ────────────────────────────────────────────────
    core_decay = [c for c in ALL_DECAY if c not in RATING_COLS]
    first = df.sort_values("date").groupby("team", sort=False).nth(0)[core_decay]
    if first.notna().any().any():
        bad = first.columns[first.notna().any()].tolist()
        print(f"ALERTA LEAKAGE: colunas com valor no primeiro jogo: {bad}")
        sys.exit(1)

    if verbose:
        n_gls  = df["gls_decay"].notna().sum()
        n_rat  = df["delta_rating_decay"].notna().sum()
        n_elo  = df["elo_diff_decay"].notna().sum()
        n_tit  = df["rating_titulares_decay"].notna().sum()
        n_has  = int(df["has_decay_data"].sum())
        print(f"PASSO 2 — Decay OK (ξ={xi:.3f})  |  Anti-leakage OK")
        print(f"          gls_decay preenchido      : {n_gls} ({n_gls/len(df)*100:.1f}%)")
        print(f"          delta_rating_decay        : {n_rat} ({n_rat/len(df)*100:.1f}%)")
        print(f"          elo_diff_decay            : {n_elo} ({n_elo/len(df)*100:.1f}%)")
        print(f"          rating_titulares_decay    : {n_tit} ({n_tit/len(df)*100:.1f}%)")
        print(f"          has_decay_data=True       : {n_has} ({n_has/len(df)*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# PASSO 2b — Imputação por mediana da confederação
# ---------------------------------------------------------------------------

def impute_by_confederation(df: pd.DataFrame, impute_ratings: bool = True,
                             verbose: bool = True) -> pd.DataFrame:
    df["imputed"] = False
    df["saves_decay_imputed"]      = False
    df["rating_titulares_imputed"] = False

    rating_coverage = df["delta_rating_decay"].notna().mean()
    use_ratings     = impute_ratings and (rating_coverage > 0.05)
    decay_cols      = ALL_DECAY if use_ratings else [c for c in ALL_DECAY if c not in RATING_COLS]

    # Use only rows with real decay data (has_decay_data=True) to compute confederation
    # medians — avoids contaminating the reference with the imputed-zero observations.
    real_mask      = df["has_decay_data"] == True
    global_medians = df.loc[real_mask, decay_cols].median()

    for conf, grp_conf in df.groupby("confederation"):
        real_conf     = grp_conf[grp_conf["has_decay_data"] == True]
        conf_medians  = real_conf[decay_cols].median()
        for col in decay_cols:
            fill_val = conf_medians[col]
            if pd.isna(fill_val):
                fill_val = global_medians[col]
            if pd.isna(fill_val):
                continue
            target = (df["confederation"] == conf) & df[col].isna()
            if target.any():
                df.loc[target, col]       = fill_val
                df.loc[target, "imputed"] = True
                if col == "saves_decay":
                    df.loc[target, "saves_decay_imputed"] = True
                elif col == "rating_titulares_decay":
                    df.loc[target, "rating_titulares_imputed"] = True

    n_imp     = int(df["imputed"].sum())
    n_sv_imp  = int(df["saves_decay_imputed"].sum())
    n_tit_imp = int(df["rating_titulares_imputed"].sum())
    if verbose:
        conf_counts = df[df["imputed"]].groupby("confederation").size().to_dict()
        print(f"PASSO 2b — Imputação por confederação: {n_imp} linhas preenchidas")
        for conf, cnt in sorted(conf_counts.items()):
            print(f"           {conf}: {cnt} linhas")
        print(f"           saves_decay imputados     : {n_sv_imp}")
        print(f"           rating_titulares imputados: {n_tit_imp}")
    return df


# ---------------------------------------------------------------------------
# PASSO 3 — Pivotar: 1 linha por jogo (home + away features)
# ---------------------------------------------------------------------------

def build_match_dataset(df: pd.DataFrame) -> pd.DataFrame:
    home = df[df["home_away"] == "home"].copy()
    away = df[df["home_away"] == "away"].copy()

    meta   = ["match_id", "date", "competition", "confederation", "match_type"]
    ha_col = ["home_advantage"] if "home_advantage" in home.columns else []
    extra  = ["has_decay_data", "saves_decay_imputed", "rating_titulares_imputed"]

    home_sel = home[meta + ["team", "opponent", "gols_marcados", "imputed"]
                    + ha_col + extra + ALL_DECAY].rename(columns={
        "team":                     "home_team",
        "opponent":                 "away_team",
        "gols_marcados":            "home_gols",
        "imputed":                  "home_imputed",
        "has_decay_data":           "home_has_decay_data",
        "saves_decay_imputed":      "home_saves_decay_imputed",
        "rating_titulares_imputed": "home_rating_titulares_imputed",
        **{c: f"home_{c}" for c in ALL_DECAY},
    })
    away_sel = away[["match_id", "gols_marcados", "imputed", "has_decay_data",
                      "saves_decay_imputed", "rating_titulares_imputed"]
                    + ALL_DECAY].rename(columns={
        "gols_marcados":            "away_gols",
        "imputed":                  "away_imputed",
        "has_decay_data":           "away_has_decay_data",
        "saves_decay_imputed":      "away_saves_decay_imputed",
        "rating_titulares_imputed": "away_rating_titulares_imputed",
        **{c: f"away_{c}" for c in ALL_DECAY},
    })

    merged = home_sel.merge(away_sel, on="match_id", how="inner")

    # ETAPA 4 — match_type_wcq mantido para compatibilidade mas não entra em FEATURES v8
    # (sinal endógeno: WCQ são exatamente os jogos que queremos prever)
    merged["match_type_wcq"] = (merged["match_type"] == "WCQ").astype(int)

    # ETAPA 4 — home_advantage renomeado para host_factor (v8)
    # Em treino (WCQ/friendlies): 0 ou 1 — indica se o time da casa joga no próprio estádio.
    # Em Copa 2026: apenas USA/México/Canadá recebem valor > 0 (calibrado pelo histórico de Copas).
    # A renomeação acontece no simulate_copa2026.py para preservar retrocompatibilidade do CSV.
    if "home_advantage" not in merged.columns:
        merged["home_advantage"] = 0

    # ETAPA 4 — conf_FRIENDLY removida: não é confederação real, introduz ruído ao modelo
    merged = pd.get_dummies(merged, columns=["confederation"], prefix="conf", dtype=int)
    if "conf_FRIENDLY" in merged.columns:
        merged.drop(columns=["conf_FRIENDLY"], inplace=True)

    merged = merged.sort_values("date").reset_index(drop=True)

    print(f"PASSO 3 — Dataset final: {len(merged)} jogos × {len(merged.columns)} colunas")
    return merged


# ---------------------------------------------------------------------------
# PASSO 4 — Validação por time (Brasil, Noruega, Japão)
# ---------------------------------------------------------------------------

def _print_decay_validation(df: pd.DataFrame, xi: float,
                             teams=("Brazil", "Norway", "Japan")) -> None:
    print(f"\n{'='*60}")
    print(f"  VALIDAÇÃO DECAY — ξ = {xi:.3f}")
    print(f"{'='*60}")

    for team in teams:
        team_df = df[df["team"] == team].sort_values("date").reset_index(drop=True)
        if team_df.empty:
            print(f"\n  {team}: sem dados")
            continue

        n_total = len(team_df)
        last    = team_df.iloc[-1]

        # Compute simple avg5 for comparison (last 5 non-NaN before last game)
        past_gls = team_df["gols_marcados"].values[:-1].astype(float)
        valid_gls = past_gls[~np.isnan(past_gls)]
        gls_avg5  = float(np.mean(valid_gls[-5:])) if len(valid_gls) > 0 else np.nan

        gls_decay_val = last["gls_decay"] if pd.notna(last.get("gls_decay")) else np.nan

        # Weight distribution for last game
        ref_date   = team_df["date"].values[-1]
        past_mask  = team_df["date"].values < ref_date
        n_past     = int(past_mask.sum())

        print(f"\n  {team}:")
        print(f"    Jogos no histórico (excl. último) : {n_past}")
        print(f"    has_decay_data (último jogo)      : {bool(last.get('has_decay_data', False))}")
        print(f"    gls_decay  (último jogo)          : {gls_decay_val:.4f}" if not np.isnan(gls_decay_val) else "    gls_decay: NaN")
        print(f"    gls_avg5   (simples, últimos 5)   : {gls_avg5:.4f}" if not np.isnan(gls_avg5) else "    gls_avg5 (simples): NaN")

        if n_past > 0:
            past_dates = team_df["date"].values[past_mask]
            past_mt    = team_df["match_type"].values[past_mask]
            past_mw    = np.where(past_mt == "WCQ", MATCH_WEIGHT_WCQ, MATCH_WEIGHT_FRIENDLY)
            days_arr   = ((ref_date - past_dates) / np.timedelta64(1, "D")).astype(float)
            raw_w      = np.exp(-xi * days_arr) * past_mw
            norm_w     = raw_w / raw_w.sum()

            print(f"    Pesos w_final: min={norm_w.min():.4f}  max={norm_w.max():.4f}  "
                  f"median={float(np.median(norm_w)):.4f}")

            # top-3 mais influentes
            top3 = np.argsort(norm_w)[-3:][::-1]
            past_sub = team_df[past_mask].reset_index(drop=True)
            print(f"    Top-3 jogos mais influentes:")
            for j in top3:
                d    = pd.Timestamp(past_dates[j]).date()
                opp  = past_sub.iloc[j]["opponent"] if "opponent" in past_sub.columns else "?"
                mt   = past_sub.iloc[j]["match_type"]
                print(f"      {d} vs {opp} [{mt}]: w_norm={norm_w[j]:.4f}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Validação BUG 1+2 — gls_ponderado vs gls_decay (v9)
# ---------------------------------------------------------------------------

def _print_ponderado_validation(df: pd.DataFrame,
                                 teams=("Brazil", "Austria", "France", "Morocco")) -> None:
    """
    Imprime tabela comparativa gls_decay vs gls_ponderado_decay para validar correção v9.
    Critério: times que jogaram contra fortes devem ter gls_ponderado > gls_decay.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("  VALIDAÇÃO v9 — gls_ponderado_decay (BUG 1+2 corrigidos)")
    print(f"  ELO_MEDIO_GLOBAL={ELO_MEDIO_GLOBAL:.0f}  |  ELO_FORTE_THRESHOLD={ELO_FORTE_THRESHOLD}")
    print(sep)
    print(f"  {'Time':15s}  {'gls_decay':>10}  {'gls_pond':>10}  {'Δ%':>8}  "
          f"{'vs_forte':>9}  {'vs_fraco':>9}  {'Status':12}")
    print("  " + "-" * 68)

    for team in teams:
        team_df = df[df["team"] == team].sort_values("date")
        if team_df.empty:
            print(f"  {team:15s}  (sem dados)")
            continue
        last = team_df.iloc[-1]
        gls   = last.get("gls_decay", np.nan)
        gls_p = last.get("gls_ponderado_decay", np.nan)
        vf    = last.get("gls_decay_vs_forte", np.nan)
        vfr   = last.get("gls_decay_vs_fraco", np.nan)

        delta_pct = ((gls_p - gls) / gls * 100) if pd.notna(gls) and gls > 0 and pd.notna(gls_p) else np.nan
        gls_s   = f"{gls:.3f}"   if pd.notna(gls)   else "  NaN"
        gls_p_s = f"{gls_p:.3f}" if pd.notna(gls_p) else "  NaN"
        dp_s    = f"{delta_pct:+.1f}%" if pd.notna(delta_pct) else "  NaN"
        vf_s    = f"{vf:.3f}"    if pd.notna(vf)    else "  NaN"
        vfr_s   = f"{vfr:.3f}"   if pd.notna(vfr)   else "  NaN"

        # Expectativa: Brasil/França/Marrocos jogam fortes → ponderado >= simples
        expected_up = team in ("Brazil", "France", "Morocco")
        if pd.notna(delta_pct):
            ok = (delta_pct >= 0) if expected_up else (delta_pct <= 0)
            status = "OK ✓" if ok else "FALHOU ✗"
        else:
            status = "NaN"

        print(f"  {team:15s}  {gls_s:>10}  {gls_p_s:>10}  {dp_s:>8}  "
              f"{vf_s:>9}  {vfr_s:>9}  {status}")

    print(sep)


# ---------------------------------------------------------------------------
# PASSO 5 — Diagnóstico final
# ---------------------------------------------------------------------------

def report_diagnostics(df_raw: pd.DataFrame, df_model: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("PASSO 4 — Diagnóstico final")
    print(f"{'='*60}")

    h_sog = df_model.get("home_shots_on_goal_decay", pd.Series(dtype=float))
    a_sog = df_model.get("away_shots_on_goal_decay", pd.Series(dtype=float))
    sog_ok       = h_sog.notna() & a_sog.notna()
    sog_ok_total = int(sog_ok.sum())
    print(f"\n  Jogos com shots_on_goal_decay completos (home E away): {sog_ok_total}")

    conf_cols = [c for c in df_model.columns if c.startswith("conf_")]
    print(f"\n  Por confederação (shots_on_goal_decay completos):")
    for cc in sorted(conf_cols):
        name  = cc.replace("conf_", "")
        mask  = df_model[cc] == 1
        total = int(mask.sum())
        with_sog = int((mask & sog_ok).sum())
        pct = with_sog / total * 100 if total else 0
        print(f"    {name:<12}: {with_sog:>4}/{total:<4} ({pct:.0f}%)")

    home_sog = set(df_model[df_model["home_shots_on_goal_decay"].notna()]["home_team"].unique())
    away_sog = set(df_model[df_model["away_shots_on_goal_decay"].notna()]["away_team"].unique())
    teams_sog = home_sog | away_sog
    copa_sog  = COPA2026_TEAMS & teams_sog
    print(f"\n  Copa 2026 — times com shots_on_goal_decay: {len(copa_sog)}/48")

    copa_without = COPA2026_TEAMS - teams_sog
    if copa_without:
        print(f"\n  Ainda sem shots_decay:")
        for t in sorted(copa_without):
            print(f"    {t}")

    home_d = [f"home_{c}" for c in ALL_DECAY]
    away_d = [f"away_{c}" for c in ALL_DECAY]
    all_ok = df_model[[c for c in home_d + away_d if c in df_model.columns]].notna().all(axis=1)
    print(f"\n  Jogos com TODOS decay preenchidos: {int(all_ok.sum())}")


# ---------------------------------------------------------------------------
# PASSO 6 — Salvar
# ---------------------------------------------------------------------------

def save_data(df: pd.DataFrame) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Try direct write; if file is locked (Excel open), save to _v6 path
    try:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", sep=",", date_format="%Y-%m-%d")
        print(f"\n  → Salvo: {OUTPUT_CSV}")
    except (PermissionError, OSError):
        alt = OUTPUT_CSV.with_stem(OUTPUT_CSV.stem + "_v6")
        df.to_csv(alt, index=False, encoding="utf-8", sep=",", date_format="%Y-%m-%d")
        print(f"\n  → (arquivo principal bloqueado) Salvo em: {alt}")
        # Patch OUTPUT_CSV to point to alt for downstream reads
        import builtins
        builtins._model_dataset_alt = str(alt)
    key_cols = (
        ["date", "home_team", "away_team", "home_gols", "away_gols",
         "home_gls_decay", "away_gls_decay",
         "home_shots_on_goal_decay", "away_shots_on_goal_decay",
         "home_win_rate_decay", "away_win_rate_decay",
         "home_elo_diff_decay", "away_elo_diff_decay",
         "match_type_wcq", "home_advantage"]
        + [c for c in df.columns if c.startswith("conf_")]
    )
    key_cols = [c for c in key_cols if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(df[key_cols].head(2).to_string(index=False))


# ---------------------------------------------------------------------------
# Wrapper conveniente para tune_xi em model_xgboost.py
# ---------------------------------------------------------------------------

def build_dataset_with_xi(xi: float = XI_DEFAULT, save: bool = False,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Constrói dataset completo com decay temporal (ξ=xi).
    Se save=True, salva em OUTPUT_CSV.
    """
    df_raw  = load_and_merge_raw()
    df_dec  = add_decay_features(df_raw, xi, verbose=verbose)
    df_imp  = impute_by_confederation(df_dec, verbose=verbose)
    df_wide = build_match_dataset(df_imp)
    if save:
        save_data(df_wide)
    return df_wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  build_features.py — dataset para modelo de placares")
    print("  Decay temporal exponencial (substitui avg5 fixo)")
    print("=" * 60 + "\n")

    df_raw  = load_and_merge_raw()
    df_dec  = add_decay_features(df_raw, XI_DEFAULT)
    _print_decay_validation(df_dec, XI_DEFAULT)
    _print_ponderado_validation(df_dec)
    df_imp  = impute_by_confederation(df_dec)
    df_wide = build_match_dataset(df_imp)
    report_diagnostics(df_raw, df_wide)
    save_data(df_wide)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
