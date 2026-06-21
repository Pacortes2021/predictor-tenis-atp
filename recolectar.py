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
ANIO_INI, ANIO_FIN = 2000, 2024  # mirror histórico (circuito principal)
# El histórico llega a 2024; para 2025-2026 usamos otro mirror que sigue activo (guardado en Git LFS,
# servido por media.githubusercontent.com). Trae Challengers, que filtramos para mantener consistencia.
REPO_REC = "ivanposinovec/ATP"
SUBDIR_REC = "tennis_atp-master"
RECIENTES = [2025, 2026]
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
    # años recientes (2025-2026) desde el mirror activo, solo circuito principal (excluye Challengers 'C')
    for y in RECIENTES:
        url = f"https://media.githubusercontent.com/media/{REPO_REC}/main/{SUBDIR_REC}/atp_matches_{y}.csv"
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"  {y}: HTTP {r.status_code}, saltado"); continue
        d = pd.read_csv(io.StringIO(r.content.decode("utf-8", "ignore")))
        if str(d.columns[0]).startswith("version https://git-lfs"):
            print(f"  {y}: puntero LFS sin resolver, saltado"); continue
        d = d[d["tourney_level"].isin(["G", "M", "A", "F", "D", "O"])]  # solo circuito principal (whitelist; descarta Challengers 'C' y partidos sin nivel)
        d = d[[c for c in COLS if c in d.columns]].copy()
        dfs.append(d)
        print(f"  {y}: {len(d)} partidos (circuito principal)")
    full = pd.concat(dfs, ignore_index=True)
    full["fecha"] = pd.to_datetime(full["tourney_date"], format="%Y%m%d", errors="coerce")
    # quitar posibles solapes entre fuentes (p.ej. United Cup arranca a fines de dic)
    full = full.drop_duplicates(subset=["tourney_date", "tourney_name", "round", "winner_name", "loser_name"])
    # ordenar cronológicamente (clave para Elo y validación temporal). Dentro de un torneo,
    # ordenar por ronda para respetar el orden real de juego.
    orden_ronda = {"R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6,
                   "RR": 1, "BR": 5}
    full["_r"] = full["round"].map(orden_ronda).fillna(0)
    full = full.sort_values(["fecha", "tourney_name", "_r"]).drop(columns="_r").reset_index(drop=True)
    full = full[full["surface"].isin(["Hard", "Clay", "Grass"])]  # descartar Carpet (extinta) y vacíos
    # excluir rondas de clasificación: el histórico 2000-2024 es solo cuadro principal, pero el mirror
    # 2025-2026 trae qualy (Q1/Q2/Q3). Quitarlas mantiene la base homogénea y evita inflar a jugadores recientes.
    full = full[~full["round"].isin(["Q1", "Q2", "Q3"])]
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
