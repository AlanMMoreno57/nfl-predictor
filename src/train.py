"""
train.py
========
Paso 3 del pipeline: entrena los modelos sobre data/processed/model_dataset.parquet
y los guarda en models/ para que predict.py los use.

Sigue la recomendación de la sección 15 del framework:
    - Logistic Regression como baseline obligatorio de referencia
    - XGBoost como modelo principal de producción (clasificación: home_win)
    - Un regresor (Ridge) para el margen de victoria (proxy de "spread")

Validación: walk-forward temporal (nunca k-fold aleatorio en datos
deportivos). Se entrena con todas las temporadas excepto la más
reciente completa, y se valida contra esa última temporada — simulando
cómo se comportaría el modelo prediciendo partidos que "aún no ha visto".

Uso:
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, mean_absolute_error, mean_squared_error,
)
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_COLUMNS = [
    "diff_epa_off_roll5",
    "diff_epa_def_roll5",
    "diff_elo",
    "diff_rest",
    "diff_injury_index",
    "div_game",
    "spread_line",
]


def load_dataset() -> pl.DataFrame:
    df = pl.read_parquet(PROCESSED_DIR / "model_dataset.parquet")
    # Solo partidos ya jugados (con resultado) sirven para entrenar/validar.
    df = df.filter(pl.col("home_score").is_not_null())
    # Filas de semana 1 de la primera temporada del dataset no tienen
    # rolling stats (correctamente, ver features.py) — se excluyen del
    # entrenamiento porque no aportan señal de forma reciente.
    df = df.drop_nulls(subset=["diff_epa_off_roll5", "diff_epa_def_roll5"])
    df = df.with_columns(pl.col("div_game").cast(pl.Int8))
    return df


def walk_forward_split(df: pl.DataFrame):
    seasons = sorted(df["season"].unique().to_list())
    test_season = seasons[-1]
    train_df = df.filter(pl.col("season") < test_season)
    test_df = df.filter(pl.col("season") == test_season)
    print(f"[train] Split walk-forward: train={seasons[0]}-{seasons[-2]} "
          f"({train_df.height} juegos) | test={test_season} ({test_df.height} juegos)")
    return train_df, test_df


def to_xy(df: pl.DataFrame):
    X = df.select(FEATURE_COLUMNS).to_pandas()
    y_class = df["home_win"].to_pandas()
    y_reg = df["home_margin"].to_pandas()
    return X, y_class, y_reg


def evaluate_classifier(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"  [{name}] accuracy={acc:.4f}  log_loss={ll:.4f}  brier={brier:.4f}")
    return {"accuracy": acc, "log_loss": ll, "brier": brier}


def evaluate_regressor(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"  [{name}] MAE={mae:.3f} puntos  RMSE={rmse:.3f} puntos")
    return {"mae": mae, "rmse": rmse}


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[train] Cargando dataset procesado...")
    df = load_dataset()

    train_df, test_df = walk_forward_split(df)
    X_train, y_train_class, y_train_reg = to_xy(train_df)
    X_test, y_test_class, y_test_reg = to_xy(test_df)

    print("[train] Entrenando Logistic Regression (baseline)...")
    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_train, y_train_class)
    metrics_logit = evaluate_classifier("LogisticRegression", logit, X_test, y_test_class)

    print("[train] Entrenando XGBoost (modelo principal)...")
    xgb = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        reg_lambda=2.0, reg_alpha=0.5, subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train_class)
    metrics_xgb = evaluate_classifier("XGBoost", xgb, X_test, y_test_class)

    print("[train] Entrenando Ridge (regresor de margen / proxy de spread)...")
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_train, y_train_reg)
    metrics_ridge = evaluate_regressor("Ridge (margen)", ridge, X_test, y_test_reg)

    # Comparación directa contra la línea de Vegas como referencia de
    # calibración (sección 16.5 del framework): ¿qué tan lejos está el
    # mercado del resultado real, para contextualizar si el modelo es útil?
    # En nflreadr, spread_line positivo = el LOCAL es favorito por esa
    # cantidad de puntos (convención distinta a la típica de casas de
    # apuestas en EE.UU., donde el favorito se muestra en negativo).
    # Confirmado empíricamente: corr(spread_line, home_margin) = +0.45.
    vegas_mae = mean_absolute_error(y_test_reg, test_df["spread_line"].to_pandas())
    print(f"  [Vegas spread_line como predictor] MAE={vegas_mae:.3f} puntos (referencia de mercado)")

    print("[train] Guardando modelos en models/ ...")
    joblib.dump(logit, MODELS_DIR / "logit_home_win.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgb_home_win.joblib")
    joblib.dump(ridge, MODELS_DIR / "ridge_home_margin.joblib")

    report = {
        "test_season": int(test_df["season"][0]),
        "n_train_games": train_df.height,
        "n_test_games": test_df.height,
        "feature_columns": FEATURE_COLUMNS,
        "logistic_regression": metrics_logit,
        "xgboost": metrics_xgb,
        "ridge_margin": metrics_ridge,
        "vegas_benchmark_mae": vegas_mae,
    }
    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[train] Listo. Reporte de métricas en models/training_report.json")


if __name__ == "__main__":
    main()
