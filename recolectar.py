"""Descarga los partidos ATP (todas las superficies) desde un mirror del dataset de
Jeff Sackmann y los guarda en un único CSV limpio.

El repo original github.com/JeffSackmann/tennis_atp se volvió privado (jun-2026); usamos
el mirror sacriusdt/tennis-atp-prediction, que conserva el mismo esquema (atp_matches_YYYY.csv).
"""
import io
import pandas as pd
import requests

REPO = "sacriusdt/tennis-atp-prediction"
BR = "main"
ANIO_INI, ANIO_FIN = 2000, 2024  # 25 temporadas
DATA = "/Users/pabloignaciocortesvielma/Downloads/Tenis_Predictor/data"

# columnas que de verdad usamos (identificación + resultado + ranking + box score de saque/resto)
COLS = [
    "tourney_date", "tourney_name", "tourney_level", "surface", "best_of", "round", "score",
    "winner_name", "loser_name", "winner_rank", "loser_rank", "winner_age", "loser_age",
    "winner_hand", "loser_hand", "winner_ht", "loser_ht",
    "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt", "w_1stIn", "l_1stIn",
    "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon", "w_bpSaved", "l_bpSaved",
    "w_bpFaced", "l_bpFaced", "w_SvGms", "l_SvGms",
]


def descargar():
    dfs = []
    for y in range(ANIO_INI, ANIO_FIN + 1):
        url = f"https://raw.githubusercontent.com/{REPO}/{BR}/atp_matches_{y}.csv"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  {y}: HTTP {r.status_code}, saltado")
            continue
        d = pd.read_csv(io.StringIO(r.content.decode("utf-8", "ignore")))
        d = d[[c for c in COLS if c in d.columns]].copy()
        dfs.append(d)
        print(f"  {y}: {len(d)} partidos")
    full = pd.concat(dfs, ignore_index=True)
    full["fecha"] = pd.to_datetime(full["tourney_date"], format="%Y%m%d", errors="coerce")
    # ordenar cronológicamente (clave para Elo y validación temporal). Dentro de un torneo,
    # ordenar por ronda para respetar el orden real de juego.
    orden_ronda = {"R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6,
                   "RR": 1, "BR": 5}
    full["_r"] = full["round"].map(orden_ronda).fillna(0)
    full = full.sort_values(["fecha", "tourney_name", "_r"]).drop(columns="_r").reset_index(drop=True)
    full = full[full["surface"].isin(["Hard", "Clay", "Grass"])]  # descartar Carpet (extinta) y vacíos
    return full


if __name__ == "__main__":
    print(f"Descargando ATP {ANIO_INI}-{ANIO_FIN} desde {REPO}...")
    df = descargar()
    out = f"{DATA}/partidos.csv"
    df.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(f"  {len(df)} partidos | superficies: {df.surface.value_counts().to_dict()}")
    print(f"  cobertura serve stats (w_svpt): {df.w_svpt.notna().mean():.0%}")
    print(f"  rango fechas: {df.fecha.min().date()} -> {df.fecha.max().date()}")
