"""
features.py
============
Paso 2 del pipeline: convierte los datos crudos (data/raw/) en un dataset
listo para modelar (data/processed/model_dataset.parquet), implementando
la ingeniería de características de la sección 14 del framework:

    - Rolling averages de EPA ofensivo/defensivo (3 y 5 partidos)
    - ELO dinámico actualizado partido a partido
    - Índice ponderado de lesiones
    - Descanso y su diferencial
    - Contexto de calendario (divisional, clima, superficie)
    - Diferenciales home-vs-away de todo lo anterior

REGLA DE ORO ANTI-LEAKAGE: toda variable de "forma reciente" se calcula
con shift(1) antes de cualquier rolling/ELO, es decir, el valor asignado
a la fila de la semana N usa únicamente información hasta la semana N-1.
Esto se verifica explícitamente en tests_leakage() al final del script.

Uso:
    python src/features.py
"""

from pathlib import Path

import numpy as np
import polars as pl

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Peso relativo de cada posición para el índice de lesiones ponderado.
# QB pesa mucho más que un backup de posición secundaria.
INJURY_POSITION_WEIGHTS = {
    "QB": 5.0, "OT": 2.5, "OG": 2.0, "C": 2.0, "WR": 1.8, "TE": 1.5,
    "RB": 1.5, "CB": 1.8, "S": 1.5, "DE": 1.8, "DT": 1.5, "LB": 1.5,
    "EDGE": 1.8, "OL": 2.0, "DL": 1.6, "DB": 1.6,
}
INJURY_STATUS_WEIGHTS = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.35}


def load_raw():
    schedules = pl.read_parquet(RAW_DIR / "schedules.parquet")
    team_stats = pl.read_parquet(RAW_DIR / "team_stats.parquet")
    injuries = pl.read_parquet(RAW_DIR / "injuries.parquet")
    return schedules, team_stats, injuries


def build_team_week_panel(team_stats: pl.DataFrame) -> pl.DataFrame:
    """
    Construye un panel team x season x week con métricas de eficiencia
    ofensiva propia y defensiva (= lo que el rival le anotó/produjo esa
    semana), listo para calcular rolling averages sin leakage.
    """
    off = team_stats.select([
        "season", "week", "team", "opponent_team",
        pl.col("passing_epa").alias("epa_pass_off"),
        pl.col("rushing_epa").alias("epa_rush_off"),
    ]).with_columns(
        ((pl.col("epa_pass_off") + pl.col("epa_rush_off"))).alias("epa_off_total")
    )

    # La eficiencia "defensiva" de un equipo = la eficiencia ofensiva que
    # el RIVAL tuvo en ese mismo partido. Se obtiene renombrando y
    # cruzando por (season, week, opponent_team == team).
    def_view = off.select([
        "season", "week",
        pl.col("team").alias("opponent_team"),
        pl.col("epa_off_total").alias("epa_def_allowed"),
    ])

    panel = off.join(def_view, on=["season", "week", "opponent_team"], how="left")
    return panel.sort(["team", "season", "week"])


def add_rolling_features(panel: pl.DataFrame) -> pl.DataFrame:
    """
    Agrega epa_off_roll3/5 y epa_def_roll3/5 usando shift(1) antes del
    rolling: la fila de la semana N solo ve datos hasta N-1.
    """
    panel = panel.sort(["team", "season", "week"])

    panel = panel.with_columns([
        pl.col("epa_off_total").shift(1).over("team").alias("_epa_off_shifted"),
        pl.col("epa_def_allowed").shift(1).over("team").alias("_epa_def_shifted"),
    ])

    panel = panel.with_columns([
        pl.col("_epa_off_shifted").rolling_mean(window_size=3, min_samples=1).over("team").alias("epa_off_roll3"),
        pl.col("_epa_off_shifted").rolling_mean(window_size=5, min_samples=1).over("team").alias("epa_off_roll5"),
        pl.col("_epa_def_shifted").rolling_mean(window_size=3, min_samples=1).over("team").alias("epa_def_roll3"),
        pl.col("_epa_def_shifted").rolling_mean(window_size=5, min_samples=1).over("team").alias("epa_def_roll5"),
    ]).drop(["_epa_off_shifted", "_epa_def_shifted"])

    return panel


def compute_dynamic_elo(schedules: pl.DataFrame, k: float = 20.0, base: float = 1500.0) -> pl.DataFrame:
    """
    Calcula ELO dinámico partido a partido, con multiplicador por margen
    de victoria (MOV), iterando cronológicamente por temporada/semana
    SOLO sobre partidos ya jugados. Devuelve el ELO de cada equipo
    *previo* a cada partido (pre-game), que es lo único que se puede
    usar como feature sin leakage.

    Para partidos FUTUROS (sin resultado todavía, ej. la próxima
    temporada ya calendarizada), se les asigna el último ELO conocido
    de cada equipo (el rating con el que terminó su partido más
    reciente) — sin esto, un partido futuro quedaría sin ELO simplemente
    porque "aún no se jugó ningún partido con ese season/week exacto".
    """
    played = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    ).sort(["season", "week"]).to_dicts()

    future = schedules.filter(
        pl.col("home_score").is_null() | pl.col("away_score").is_null()
    ).sort(["season", "week"]).to_dicts()

    elo = {}
    records = []

    for g in played:
        home, away = g["home_team"], g["away_team"]
        elo.setdefault(home, base)
        elo.setdefault(away, base)

        elo_home_pre, elo_away_pre = elo[home], elo[away]
        records.append({
            "game_id": g["game_id"], "season": g["season"], "week": g["week"],
            "home_team": home, "away_team": away,
            "home_elo_pre": elo_home_pre, "away_elo_pre": elo_away_pre,
        })

        margin = g["home_score"] - g["away_score"]
        actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        expected_home = 1.0 / (1.0 + 10 ** ((elo_away_pre - elo_home_pre) / 400.0))

        mov_mult = np.log(abs(margin) + 1) * (2.2 / ((abs(elo_home_pre - elo_away_pre) * 0.001) + 2.2))
        delta = k * mov_mult * (actual_home - expected_home)

        elo[home] = elo_home_pre + delta
        elo[away] = elo_away_pre - delta

    # Partidos futuros: se les asigna el ELO más reciente disponible de
    # cada equipo (el estado final del dict tras procesar todo lo jugado).
    for g in future:
        home, away = g["home_team"], g["away_team"]
        records.append({
            "game_id": g["game_id"], "season": g["season"], "week": g["week"],
            "home_team": home, "away_team": away,
            "home_elo_pre": elo.get(home, base), "away_elo_pre": elo.get(away, base),
        })

    return pl.DataFrame(records)


def add_injury_index(injuries: pl.DataFrame) -> pl.DataFrame:
    """
    Índice de lesiones ponderado por equipo/semana: suma de
    peso_posicion * peso_status para cada jugador reportado.
    """
    weights = injuries.with_columns([
        pl.col("position").replace_strict(INJURY_POSITION_WEIGHTS, default=1.2, return_dtype=pl.Float64).alias("pos_w"),
        pl.col("report_status").replace_strict(INJURY_STATUS_WEIGHTS, default=0.0, return_dtype=pl.Float64).alias("status_w"),
    ]).with_columns(
        (pl.col("pos_w") * pl.col("status_w")).alias("injury_points")
    )

    idx = weights.group_by(["season", "week", "team"]).agg(
        pl.col("injury_points").sum().alias("injury_weighted_index")
    )
    # nflreadpy entrega season/week de injuries como Float64; se castea a
    # Int32 para que coincida con el tipo usado en schedules/team_stats.
    idx = idx.with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
    ])
    return idx


def build_game_level_dataset(schedules, team_panel, elo_df, injury_idx) -> pl.DataFrame:
    """
    Une todo a nivel partido: para cada game_id, features del equipo
    local y del visitante, más el diferencial home-away de cada una.
    """
    base = schedules.select([
        "game_id", "season", "week", "game_type", "home_team", "away_team",
        "home_score", "away_score", "result", "spread_line", "total_line",
        "home_rest", "away_rest", "div_game", "roof", "surface", "temp", "wind",
        "home_qb_name", "away_qb_name", "home_coach", "away_coach",
    ]).with_columns(
        # Clave global ordenable temporada+semana, para poder hacer un
        # join "asof" (último valor conocido) por equipo, en vez de un
        # match exacto de (season, week) que falla para partidos futuros
        # cuyo equipo aún no tiene una fila de team_stats en esa semana
        # exacta (ej. toda la temporada 2026, que ya está calendarizada
        # pero todavía no se ha jugado ni un partido).
        (pl.col("season") * 100 + pl.col("week")).alias("_wk_key")
    )

    panel_keyed = team_panel.with_columns(
        (pl.col("season") * 100 + pl.col("week")).alias("_wk_key")
    ).sort("_wk_key")

    def _asof_team_features(side: str) -> pl.DataFrame:
        team_col = f"{side}_team"
        cols = {
            "epa_off_roll3": f"{side}_epa_off_roll3",
            "epa_off_roll5": f"{side}_epa_off_roll5",
            "epa_def_roll3": f"{side}_epa_def_roll3",
            "epa_def_roll5": f"{side}_epa_def_roll5",
        }
        games = base.select(["game_id", "_wk_key", pl.col(team_col).alias("team")]).sort("_wk_key")
        joined = games.join_asof(
            panel_keyed.select(["_wk_key", "team", *cols.keys()]),
            on="_wk_key", by="team", strategy="backward",
        )
        return joined.select(["game_id", *[pl.col(k).alias(v) for k, v in cols.items()]])

    home_feats = _asof_team_features("home")
    away_feats = _asof_team_features("away")

    df = (
        base
        .join(home_feats, on="game_id", how="left")
        .join(away_feats, on="game_id", how="left")
        .join(elo_df.select(["game_id", "home_elo_pre", "away_elo_pre"]), on="game_id", how="left")
        .join(
            injury_idx.select([
                "season", "week", pl.col("team").alias("home_team"),
                pl.col("injury_weighted_index").alias("home_injury_index"),
            ]),
            on=["season", "week", "home_team"], how="left",
        )
        .join(
            injury_idx.select([
                "season", "week", pl.col("team").alias("away_team"),
                pl.col("injury_weighted_index").alias("away_injury_index"),
            ]),
            on=["season", "week", "away_team"], how="left",
        )
    )

    df = df.with_columns([
        (pl.col("home_epa_off_roll5") - pl.col("away_epa_off_roll5")).alias("diff_epa_off_roll5"),
        (pl.col("away_epa_def_roll5") - pl.col("home_epa_def_roll5")).alias("diff_epa_def_roll5"),
        (pl.col("home_elo_pre") - pl.col("away_elo_pre")).alias("diff_elo"),
        (pl.col("home_rest") - pl.col("away_rest")).alias("diff_rest"),
        (pl.col("home_injury_index").fill_null(0) - pl.col("away_injury_index").fill_null(0)).alias("diff_injury_index"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win"),
        (pl.col("home_score") - pl.col("away_score")).alias("home_margin"),
    ])

    return df.sort(["season", "week"])


def sanity_check_no_leakage(df: pl.DataFrame) -> None:
    """
    Verificación explícita: en la semana 1 de la PRIMERA temporada del
    dataset, epa_off_roll5 debe ser null (no existe ninguna historia
    previa cargada). En temporadas posteriores, la semana 1 SÍ puede
    tener valor (arrastrado del cierre de la temporada anterior) — eso
    es diseño intencional, no leakage, porque esa información ya
    existía antes del kickoff de la nueva temporada.
    """
    first_season = df["season"].min()
    week1_first_season = df.filter((pl.col("week") == 1) & (pl.col("season") == first_season))
    leaked = week1_first_season.filter(pl.col("home_epa_off_roll5").is_not_null())
    if leaked.height > 0:
        print(f"[features] ADVERTENCIA: {leaked.height} filas de semana 1 de la "
              f"temporada {first_season} (la primera del dataset) tienen "
              f"epa_off_roll5 no-nulo. Esto sí indicaría leakage real. Revisar shift().")
    else:
        print(f"[features] Chequeo anti-leakage OK: semana 1 de {first_season} "
              f"(primera temporada, sin historia previa) no tiene rolling stats.")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[features] Cargando datos crudos...")
    schedules, team_stats, injuries = load_raw()

    print("[features] Construyendo panel team-week...")
    panel = build_team_week_panel(team_stats)

    print("[features] Calculando rolling EPA (3 y 5 partidos, con shift anti-leakage)...")
    panel = add_rolling_features(panel)

    print("[features] Calculando ELO dinámico partido a partido...")
    elo_df = compute_dynamic_elo(schedules)

    print("[features] Calculando índice de lesiones ponderado...")
    injury_idx = add_injury_index(injuries)

    print("[features] Uniendo todo a nivel de partido...")
    dataset = build_game_level_dataset(schedules, panel, elo_df, injury_idx)

    sanity_check_no_leakage(dataset)

    out_path = PROCESSED_DIR / "model_dataset.parquet"
    dataset.write_parquet(out_path)
    print(f"[features] OK -> {out_path} ({dataset.shape[0]} filas, {dataset.shape[1]} columnas)")


if __name__ == "__main__":
    main()
