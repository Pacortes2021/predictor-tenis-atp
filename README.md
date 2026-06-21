# 🎾 Predictor de Tenis ATP (todas las superficies)

Reconstrucción rigurosa de un trabajo previo de predicción de Wimbledon (solo pasto),
ahora para **Hard / Clay / Grass**, con la metodología aplicada en los proyectos de fútbol.

## Qué hace
- Predice el ganador de un partido ATP en cualquier superficie, con probabilidad y cuota justa.
- Rankings Elo general y **por superficie** (Nadal #1 en tierra, Djokovic #1 en pasto…).
- Simulador Monte Carlo de un cuadro (4/8/16 jugadores) → P(campeón).

## Datos (híbrido, ~todas las superficies, 2000 → hoy)
Fuente en tres capas, todas en formato Sackmann y **solo cuadro principal** (sin Challengers ni qualy):
1. **Histórico 2000–2024**: mirror `sacriusdt/tennis-atp-prediction` (el repo original `JeffSackmann/tennis_atp` se volvió privado en 2026).
2. **2025–mar 2026**: mirror activo `ivanposinovec/ATP` (en Git LFS, vía `media.githubusercontent.com`).
3. **abr 2026 → hoy (en vivo)**: API pública de ESPN (`recolectar_espn.py`), con reconciliación de nombres
   (normaliza acentos/guiones a la grafía del histórico) y de superficie (clasificador por torneo).
   ESPN no trae stats de saque (se dejan vacías; no afectan al modelo de resultado) y el ranking se toma
   del endpoint de rankings de ESPN.

```bash
python3 recolectar.py        # base histórica (capas 1+2) -> data/partidos.csv
python3 recolectar_espn.py   # añade la capa viva de ESPN hasta hoy (re-ejecutable para actualizar)
python3 analisis.py          # análisis de variables + validación + métricas
streamlit run app_tenis.py
```

## Modelo
- **Elo por superficie** con K decreciente (método Sackmann/538): general + Hard/Clay/Grass.
- Features point-in-time (walk-forward, sin fuga de futuro).
- Regresión logística **sin intercepto** (antisimetría f(A,B) = −f(B,A)), 1 fila por partido
  con orientación aleatoria (en vez de duplicar filas, que con KFold shuffle filtraba el espejo).
- Variables elegidas por **selección forward** (CV temporal, log-loss):
  `elo_general + elo_superficie + edad + log_ranking`. Las stats de saque/resto no aportan
  por encima de esto (Lasso las mantiene pero solo suman 0.001 de log-loss).
- **Validación temporal** (train <2023, test 2023–2024): log-loss 0.621, AUC 0.709, acierto 64.4%.

## Mejoras sobre el trabajo original
| Tema | Original | Esta versión |
|---|---|---|
| Fuerza | `rank_diff` | Elo general + Elo por superficie |
| Superficies | Solo pasto | Hard / Clay / Grass |
| Validación | KFold aleatorio (serie temporal) | Temporal pasado→futuro |
| Selección | Por AUC (y por 0.0001) | Forward por log-loss + calibración |
| Simulación | Determinista (favorito siempre avanza) | Monte Carlo (5.000 sorteos) |

## Deploy en Streamlit Cloud
Subir esta carpeta a un repo de GitHub y apuntar la app a `app_tenis.py`.
`requirements.txt` ya cubre todo. Los CSV de `data/` van commiteados (se actualizan con `recolectar.py`).
