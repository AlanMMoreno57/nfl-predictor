# NFL Predictor — Pipeline automatizado de predicción de partidos

Sistema de Machine Learning que descarga datos NFL actualizados, entrena un modelo,
genera predicciones semanales, y las muestra en un dashboard web — todo automatizado
con GitHub Actions (gratis, sin servidores).

## Cómo funciona el ciclo completo

```
┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌─────────────┐    ┌───────────┐
│  ingest.py   │ -> │ features.py  │ -> │ train.py  │ -> │ predict.py  │ -> │ dashboard │
│ descarga     │    │ EPA rolling, │    │ Logistic +│    │ genera JSON │    │ GitHub    │
│ nflverse     │    │ ELO, lesiones│    │ XGBoost   │    │ de la semana│    │ Pages     │
└─────────────┘    └──────────────┘    └───────────┘    └─────────────┘    └───────────┘
```

Cada **martes a las 9:00 UTC**, `.github/workflows/weekly_update.yml` corre este ciclo
completo automáticamente: descarga los resultados más recientes, reconstruye las
features, reentrena el modelo con los datos frescos, genera las predicciones de la
próxima semana, y publica el dashboard actualizado en GitHub Pages. No necesitas hacer
nada manualmente una vez configurado.

## Fuente de datos

Todo viene de [nflverse](https://github.com/nflverse) vía la librería `nflreadpy`
(gratuita, pública, la misma fuente que alimenta nflfastR/EPA):

- **schedules**: calendario, resultados, líneas de Vegas, clima, descanso, QB/coach por partido
- **team_stats**: EPA ofensivo/defensivo y ~130 métricas más, por equipo y semana
- **injuries**: reportes oficiales de lesiones semanales

Importante: el calendario de la temporada siguiente suele publicarse con meses de
antelación (ej. en agosto 2026 ya existe el calendario completo de 2026-27), aunque
`team_stats`/`injuries` de esa temporada no existan hasta que se jueguen partidos.
El pipeline maneja esto automáticamente (ver comentarios en `ingest.py`).

## Metodología del modelo

- **Features**: EPA rolling (3/5 partidos, con `shift(1)` anti-leakage), ELO dinámico
  con ajuste por margen de victoria, índice de lesiones ponderado por posición e
  importancia del jugador, descanso, contexto divisional, y la línea de Vegas como
  feature adicional.
- **Modelos**: Logistic Regression (baseline) + XGBoost (principal) para
  victoria/derrota, promediados en un ensemble simple; Ridge para el margen de puntos.
- **Validación**: walk-forward temporal (se entrena con todas las temporadas menos la
  más reciente completa, y se valida contra esa última temporada — nunca k-fold
  aleatorio en datos deportivos, porque mezclaría pasado y futuro).
- **Resultados de referencia** (holdout temporada 2025): ~66% de accuracy en
  victoria/derrota, MAE de ~9.7 puntos en margen — prácticamente empatado con la línea
  de Vegas (MAE ~9.7 puntos), lo cual es el resultado honesto esperado: el modelo no
  pretende "vencer" al mercado de forma sistemática, sino igualarlo con un sistema
  propio, interpretable y auditable (ver sección 16.5 del framework de ML).

Para el catálogo completo de variables, ingeniería de características y justificación
de cada decisión metodológica, ver el documento `Framework_ML_NFL_2026-2027.pdf`
entregado previamente — este repo es la implementación de ese framework.

## Estructura del repositorio

```
nfl-predictor/
├── .github/workflows/weekly_update.yml   # automatización (cron semanal)
├── data/
│   ├── raw/            # schedules.parquet, team_stats.parquet, injuries.parquet
│   └── processed/      # model_dataset.parquet (features listas para modelar)
├── models/              # modelos entrenados (.joblib) + training_report.json
├── src/
│   ├── ingest.py        # Paso 1: descarga datos crudos
│   ├── features.py      # Paso 2: ingeniería de características
│   ├── train.py         # Paso 3: entrena y evalúa los modelos
│   └── predict.py       # Paso 4: genera predictions.json de la próxima semana
├── dashboard/
│   ├── index.html       # dashboard estático (sin build step, JS vanilla)
│   └── data/predictions.json
└── requirements.txt
```

## Puesta en marcha (una sola vez)

### 1. Crear el repositorio en GitHub

```bash
cd nfl-predictor
git init
git add .
git commit -m "Setup inicial del pipeline NFL Predictor"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/nfl-predictor.git
git push -u origin main
```

### 2. Activar GitHub Pages

En el repo de GitHub: **Settings → Pages → Source → GitHub Actions** (no "Deploy from a
branch"). El workflow ya incluye los pasos de `configure-pages` / `deploy-pages`, así
que no necesitas configurar nada más — Pages se activará solo la primera vez que corra
el workflow.

### 3. Permisos del workflow

En **Settings → Actions → General → Workflow permissions**, selecciona
**"Read and write permissions"**. Esto es necesario para que el workflow pueda
commitear los datos y predicciones actualizadas cada semana.

### 4. Primera corrida

Ve a la pestaña **Actions** del repo → selecciona "Actualización semanal NFL
Predictor" → **Run workflow** (botón manual, no hace falta esperar al cron). Tras
unos minutos, tu dashboard estará en:

```
https://TU_USUARIO.github.io/nfl-predictor/
```

Después de esta primera corrida manual, el cron de los martes se encarga solo.

## Correrlo en tu computadora (sin GitHub Actions)

```bash
pip install -r requirements.txt
python src/ingest.py --start-season 2016
python src/features.py
python src/train.py
python src/predict.py
# abre dashboard/index.html en tu navegador, o:
cd dashboard && python3 -m http.server 8000
```

## Limitaciones honestas (léelas antes de apostar dinero real con esto)

- El modelo actual usa un conjunto de features deliberadamente compacto y auditable
  (7 variables). El framework completo (`Framework_ML_NFL_2026-2027.pdf`) documenta
  decenas de variables adicionales (pass block win rate, success rate, DVOA, PFF
  grades) que requieren fuentes de pago (PFF) o procesamiento adicional de
  play-by-play — quedan como extensión natural, no están implementadas aquí todavía.
- El modelo iguala aproximadamente a Vegas en margen de puntos porque usa
  `spread_line` como una de sus features — esto es intencional y honesto (sección
  16.5 del framework), no un indicador de que "vence" al mercado.
- Accuracy ~66% en victoria/derrota es consistente con lo que reportan
  públicamente proyectos similares de predicción NFL; no es una garantía para
  temporadas futuras, especialmente si hay cambios de reglas o de nivel de talento
  no capturados en el histórico de entrenamiento.
- Este proyecto es una herramienta de análisis, no asesoría financiera. Cualquier
  uso relacionado con apuestas es responsabilidad exclusiva de quien lo use.
