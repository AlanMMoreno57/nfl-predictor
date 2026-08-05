"""
predict.py
==========
Paso 4 del pipeline: genera las predicciones de la(s) próxima(s)
semana(s) usando los modelos entrenados, y escribe un JSON que el
dashboard estático consume directamente (dashboard/data/predictions.json).

Por defecto predice la PRÓXIMA semana sin resultados registrados
(la primera semana "abierta" que encuentra en el dataset). Se puede
pedir una semana/temporada específica con --season / --week.

Uso:
    python src/predict.py
    python src/predict.py --season 2026 --week 1
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import polars as pl


def _clean(value):
    """
    Convierte NaN/NaT a None. Necesario porque pandas/polars usan NaN
    para valores faltantes (ej. QB o clima aún no confirmados para
    partidos lejanos en el calendario), pero JSON estándar NO admite
    NaN como token válido — json.dumps lo escribiría igual (Python es
    permisivo), produciendo un archivo que rompe JSON.parse() en el
    navegador. Todo el output pasa por aquí antes de escribirse.
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"

FEATURE_COLUMNS = [
    "diff_epa_off_roll5",
    "diff_epa_def_roll5",
    "diff_elo",
    "diff_rest",
    "diff_injury_index",
    "div_game",
    "spread_line",
]


def find_next_open_week(df: pl.DataFrame) -> tuple[int, int]:
    """Encuentra la primera (season, week) con partidos sin resultado."""
    upcoming = (
        df.filter(pl.col("home_score").is_null())
        .select(["season", "week"])
        .unique()
        .sort(["season", "week"])
    )
    if upcoming.height == 0:
        raise RuntimeError("No hay partidos futuros sin resultado en el dataset. "
                           "¿Ya se jugó toda la temporada calendarizada? Corre ingest.py de nuevo.")
    row = upcoming.row(0)
    return int(row[0]), int(row[1])


def confidence_label(prob_home_win: float) -> str:
    edge = abs(prob_home_win - 0.5)
    if edge >= 0.20:
        return "Alta"
    elif edge >= 0.10:
        return "Media"
    return "Baja"


def main():
    parser = argparse.ArgumentParser(description="Genera predicciones para una semana NFL")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[predict] Cargando dataset y modelos...")
    df = pl.read_parquet(PROCESSED_DIR / "model_dataset.parquet")
    df = df.with_columns(pl.col("div_game").cast(pl.Int8).fill_null(0))
    df = df.with_columns([
        pl.col(c).fill_null(0.0) for c in
        ["diff_epa_off_roll5", "diff_epa_def_roll5", "diff_elo", "diff_rest", "diff_injury_index"]
    ])

    logit = joblib.load(MODELS_DIR / "logit_home_win.joblib")
    xgb = joblib.load(MODELS_DIR / "xgb_home_win.joblib")
    ridge = joblib.load(MODELS_DIR / "ridge_home_margin.joblib")

    with open(MODELS_DIR / "training_report.json") as f:
        training_report = json.load(f)

    if args.season is not None and args.week is not None:
        season, week = args.season, args.week
    else:
        season, week = find_next_open_week(df)

    print(f"[predict] Generando predicciones para temporada {season}, semana {week}...")
    games = df.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("game_id")

    if games.height == 0:
        raise RuntimeError(f"No se encontraron partidos para season={season}, week={week}.")

    X = games.select(FEATURE_COLUMNS).to_pandas()
    prob_xgb = xgb.predict_proba(X)[:, 1]
    prob_logit = logit.predict_proba(X)[:, 1]
    # Predicción final = promedio simple XGBoost + Logistic (voting
    # ensemble ligero, sección 15: reduce varianza combinando ambos).
    prob_ensemble = (prob_xgb + prob_logit) / 2.0
    pred_margin = ridge.predict(X)

    games_out = games.select([
        "game_id", "season", "week", "home_team", "away_team",
        "spread_line", "total_line", "div_game", "home_qb_name", "away_qb_name",
        "home_coach", "away_coach", "roof", "surface", "temp", "wind",
    ]).to_pandas()

    predictions = []
    for i, row in games_out.iterrows():
        p_home = float(prob_ensemble[i])
        margin = float(pred_margin[i])
        winner = row["home_team"] if p_home >= 0.5 else row["away_team"]
        predictions.append({
            "game_id": row["game_id"],
            "season": int(row["season"]),
            "week": int(row["week"]),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "predicted_winner": winner,
            "home_win_probability": round(p_home, 4),
            "away_win_probability": round(1 - p_home, 4),
            "predicted_margin_home": round(margin, 1),
            "confidence": confidence_label(p_home),
            "vegas_spread_line": _clean(row["spread_line"]),
            "vegas_total_line": _clean(row["total_line"]),
            "is_divisional": bool(row["div_game"]),
            "home_qb": _clean(row["home_qb_name"]),
            "away_qb": _clean(row["away_qb_name"]),
            "home_coach": _clean(row["home_coach"]),
            "away_coach": _clean(row["away_coach"]),
            "roof": _clean(row["roof"]),
            "surface": _clean(row["surface"]),
            "temp_f": _clean(row["temp"]),
            "wind_mph": _clean(row["wind"]),
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "week": week,
        "model_version": "xgboost+logistic_ensemble_v1",
        "training_report_summary": {
            "test_season": training_report["test_season"],
            "xgboost_accuracy": training_report["xgboost"]["accuracy"],
            "logistic_accuracy": training_report["logistic_regression"]["accuracy"],
            "vegas_benchmark_mae": training_report["vegas_benchmark_mae"],
        },
        "games": predictions,
    }

    out_path = DASHBOARD_DATA_DIR / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[predict] OK -> {out_path} ({len(predictions)} partidos)")
    for p in predictions:
        print(f"  {p['away_team']} @ {p['home_team']}: "
              f"{p['predicted_winner']} gana ({p['home_win_probability']*100:.1f}% local) "
              f"| margen esperado local: {p['predicted_margin_home']:+.1f} "
              f"| Vegas: {p['vegas_spread_line']:+.1f} | confianza: {p['confidence']}")


if __name__ == "__main__":
    main()
