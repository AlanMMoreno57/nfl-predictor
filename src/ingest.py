"""
ingest.py
=========
Paso 1 del pipeline: descarga los datos crudos necesarios para el modelo
de predicción NFL, directamente desde nflverse (fuente pública, gratuita,
la misma que alimenta nflfastR / EPA / success rate).

Este script es idempotente: se puede correr cada semana y siempre baja
la versión más reciente de los datos. Se ejecuta automáticamente vía
GitHub Actions (ver .github/workflows/weekly_update.yml), pero también
se puede correr a mano:

    python src/ingest.py

Salida (todo en data/raw/, formato parquet):
    - schedules.parquet   -> calendario, resultados, líneas de Vegas, clima, descanso
    - team_stats.parquet  -> stats ofensivas/defensivas por equipo y semana (incluye EPA)
    - injuries.parquet    -> reportes de lesiones semanales oficiales
"""

import argparse
import sys
from pathlib import Path

import nflreadpy as nfl

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_season_range(start_season: int = 2016) -> list[int]:
    """
    Temporadas con datos JUGADOS (team_stats, injuries): desde
    `start_season` hasta la temporada actual detectada por nflreadpy.
    """
    current = nfl.get_current_season()
    return list(range(start_season, current + 1))


def get_schedule_season_range(start_season: int = 2016) -> list[int]:
    """
    Temporadas para schedules: igual que arriba, MÁS la temporada
    siguiente. El calendario de la próxima temporada (fechas, rivales,
    división) suele publicarse con meses de antelación, aunque todavía
    no se haya jugado ni un partido (ej. en agosto 2026 ya existe el
    calendario completo de 2026-27). Esto permite que predict.py genere
    predicciones para la semana 1 de la nueva temporada usando la forma
    reciente (rolling stats) del cierre de la temporada anterior.
    """
    current = nfl.get_current_season()
    return list(range(start_season, current + 2))


def ingest_schedules(seasons: list[int]) -> None:
    print(f"[ingest] Descargando schedules para temporadas {seasons[0]}-{seasons[-1]}...")
    df = nfl.load_schedules(seasons=seasons)
    out_path = RAW_DIR / "schedules.parquet"
    df.write_parquet(out_path)
    print(f"[ingest] OK -> {out_path} ({df.shape[0]} filas, {df.shape[1]} columnas)")


def ingest_team_stats(seasons: list[int]) -> None:
    print(f"[ingest] Descargando team_stats para temporadas {seasons[0]}-{seasons[-1]}...")
    df = nfl.load_team_stats(seasons=seasons)
    out_path = RAW_DIR / "team_stats.parquet"
    df.write_parquet(out_path)
    print(f"[ingest] OK -> {out_path} ({df.shape[0]} filas, {df.shape[1]} columnas)")


def ingest_injuries(seasons: list[int]) -> None:
    print(f"[ingest] Descargando injuries para temporadas {seasons[0]}-{seasons[-1]}...")
    df = nfl.load_injuries(seasons=seasons)
    out_path = RAW_DIR / "injuries.parquet"
    df.write_parquet(out_path)
    print(f"[ingest] OK -> {out_path} ({df.shape[0]} filas, {df.shape[1]} columnas)")


def main():
    parser = argparse.ArgumentParser(description="Descarga datos crudos de nflverse")
    parser.add_argument(
        "--start-season",
        type=int,
        default=2016,
        help="Primera temporada histórica a incluir (default: 2016, ~10 temporadas de historia)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    seasons = get_season_range(args.start_season)
    schedule_seasons = get_schedule_season_range(args.start_season)
    print(f"[ingest] Temporada actual (con datos jugados): {seasons[-1]}")
    print(f"[ingest] Temporada siguiente (calendario, sin jugar): {schedule_seasons[-1]}")

    try:
        try:
            ingest_schedules(schedule_seasons)
        except Exception as e:
            # El calendario de la próxima temporada aún podría no estar
            # publicado (ej. muy temprano en el offseason). Se cae de
            # vuelta a las temporadas con datos confirmados.
            print(f"[ingest] Aviso: no se pudo incluir la temporada "
                  f"{schedule_seasons[-1]} ({e}). Usando solo temporadas jugadas.")
            ingest_schedules(seasons)
        ingest_team_stats(seasons)
        ingest_injuries(seasons)
    except Exception as e:
        print(f"[ingest] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("[ingest] Listo. Datos crudos actualizados en data/raw/")


if __name__ == "__main__":
    main()
