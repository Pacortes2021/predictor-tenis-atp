"""Capa VIVA del híbrido: completa partidos.csv con datos recientes de la API pública de ESPN
(desde el último partido del histórico Sackmann hasta hoy), reconciliando nombres y superficie.

ESPN no trae stats de saque (se dejan vacías; no afectan al modelo de resultado), ni edad/ranking
en el partido: el ranking se toma del endpoint de rankings de ESPN y la edad se extrapola del
histórico. Re-ejecutable: solo añade partidos posteriores al máximo ya guardado.
"""
import io
import os
import time
import unicodedata
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# ruta relativa al proyecto (portable: funciona igual en local y en Streamlit Cloud)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
H = {"User-Agent": "Mozilla/5.0"}
BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp"


def norm(s):
    """Normaliza un nombre para emparejar fuentes: sin acentos, minúsculas, guiones/puntos→espacio."""
    if not isinstance(s, str):
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = s.lower().replace("-", " ").replace(".", " ").replace("'", "")
    return " ".join(s.split())


# keywords ya normalizados (sin acentos, minúsculas); incluyen nombres de patrocinador habituales
GRASS = ["halle", "terra wortmann", "queen", "hsbc championship", "boss open", "stuttgart",
         "libema", "hertogenbosch", "rosmalen", "eastbourne", "rothesay", "nottingham", "surbiton",
         "mallorca", "newport", "hall of fame", "wimbledon", "grass"]
CLAY = ["monte carlo", "madrid", "rome", "italia", "roland garros", "french open", "barcelona",
        "munich", "bmw", "hamburg", "geneva", "gonet", "bucharest", "tiriac", "kitzbuhel", "generali",
        "bastad", "nordea", "gstaad", "swiss open", "umag", "plava laguna", "cordoba", "buenos aires",
        "rio", "santiago", "marrakech", "hassan", "houston", "sarofim", "estoril", "sardegna", "clay"]
SLAM = ["australian open", "roland garros", "french open", "wimbledon", "us open"]
MASTERS = ["indian wells", "miami", "monte carlo", "madrid", "rome", "italia",
           "canada", "toronto", "montreal", "cincinnati", "shanghai", "paris"]


def superficie(nombre):
    n = norm(nombre)  # normaliza acentos/guiones para que "Libéma" o "Monte-Carlo" emparejen
    if any(k in n for k in GRASS):
        return "Grass"
    if any(k in n for k in CLAY):
        return "Clay"
    return "Hard"


def nivel(nombre):
    n = norm(nombre)
    if any(k in n for k in SLAM):
        return "G"
    if any(k in n for k in MASTERS):
        return "M"
    return "A"


def ronda(nombre, slam=False):
    n = (nombre or "").lower()
    if "qualif" in n:
        return None  # se descarta (el histórico es solo cuadro principal)
    if n == "final" or n.endswith(" final"):
        return "F"
    if "sem" in n:
        return "SF"
    if "quarter" in n:
        return "QF"
    if "round of 128" in n:
        return "R128"
    if "round of 64" in n:
        return "R64"
    if "round of 32" in n:
        return "R32"
    if "round of 16" in n or "4th" in n:
        return "R16"
    # Slams: ESPN nombra "Round 1..4" (cuadro de 128) -> mapear al tamaño real
    if slam:
        for k, r in (("round 1", "R128"), ("round 2", "R64"), ("round 3", "R32"), ("round 4", "R16")):
            if k in n:
                return r
    return "R32"  # genérico para "Round 1/2/3" en torneos chicos (el modelo no usa la ronda)


# Grafías de ESPN que NO coinciden con el histórico Sackmann ni se salvan con norm()
# (transliteraciones distintas). Sin esto el jugador queda PARTIDO en dos identidades:
# Elo desconectado y "últimos partidos" repartidos entre dos nombres.
ALIAS = {
    "alexandr shevchenko": "Alexander Shevchenko",
    "aleksandr shevchenko": "Alexander Shevchenko",
    "abdullah shelbayh": "Abedallah Shelbayh",
    "soonwoo kwon": "Soon Woo Kwon",
}


def rankings_actuales():
    """name(norm) -> ranking ATP actual, desde el endpoint de rankings de ESPN."""
    try:
        j = requests.get(f"{BASE}/rankings", headers=H, timeout=20).json()
        ranks = j.get("rankings", [{}])[0].get("ranks", [])
        return {norm(e["athlete"]["displayName"]): e.get("current") for e in ranks if e.get("athlete")}
    except Exception:
        return {}


def main():
    base = pd.read_csv(f"{DATA}/partidos.csv", parse_dates=["fecha"])
    cols = list(base.columns)
    corte = base["fecha"].max()
    # OJO: re-barrer desde ANTES del corte. Si la corrida anterior fue a mitad de un torneo,
    # los partidos de ese mismo día que aún no terminaban (y las rondas siguientes con fecha
    # igual al corte) se perderían para siempre con un corte estricto. El dedup por clave
    # (fecha, torneo, ronda, ganador, perdedor) evita duplicar lo ya guardado.
    desde = corte - timedelta(days=7)
    print(f"Histórico hasta {corte.date()}. Re-barriendo ESPN desde {desde.date()} hasta hoy...")

    # mapas de reconciliación a partir del histórico
    canon = {}                       # nombre normalizado -> nombre canónico (Sackmann)
    edad_ult = {}                    # nombre canónico -> (última edad, última fecha)
    for _, r in base[["winner_name", "winner_age", "fecha"]].dropna(subset=["winner_name"]).iterrows():
        canon.setdefault(norm(r["winner_name"]), r["winner_name"])
    for _, r in base[["loser_name"]].dropna().iterrows():
        canon.setdefault(norm(r["loser_name"]), r["loser_name"])
    for col_n, col_a in [("winner_name", "winner_age"), ("loser_name", "loser_age")]:
        sub = base[[col_n, col_a, "fecha"]].dropna(subset=[col_n]).sort_values("fecha")
        for _, r in sub.iterrows():
            if pd.notna(r[col_a]):
                edad_ult[r[col_n]] = (r[col_a], r["fecha"])

    rk = rankings_actuales()
    print(f"  rankings ESPN cargados: {len(rk)} jugadores")

    def resolver(nombre_espn):
        nn = norm(nombre_espn)
        if nn in ALIAS:
            return ALIAS[nn]                      # transliteración conocida -> nombre del histórico
        if nn in canon:
            return canon[nn]                      # jugador conocido -> nombre del histórico (continuidad Elo)
        # jugador nuevo: nombre limpio estilo Sackmann (sin guiones) y registrar
        limpio = " ".join(w.capitalize() for w in nombre_espn.replace("-", " ").split())
        canon[nn] = limpio
        return limpio

    def edad(nombre_canon, fecha):
        if nombre_canon in edad_ult:
            a0, f0 = edad_ult[nombre_canon]
            return round(a0 + (fecha - f0).days / 365.25, 1)
        return np.nan

    # barrer fechas (cada 3 días captura todos los torneos; un evento trae su cuadro completo)
    filas = []
    vistos = set()
    d = desde.date()
    hoy = datetime.now().date()
    while d <= hoy:
        url = f"{BASE}/scoreboard?dates={d.strftime('%Y%m%d')}"
        try:
            j = requests.get(url, headers=H, timeout=25).json()
        except Exception:
            d += timedelta(days=3); continue
        for e in j.get("events", []):
            tname = e.get("name", "")
            surf, lvl = superficie(tname), nivel(tname)
            for g in e.get("groupings", []):
                if g.get("grouping", {}).get("displayName", "") != "Men's Singles":
                    continue
                for c in g.get("competitions", []):
                    if not c.get("status", {}).get("type", {}).get("completed"):
                        continue
                    rd = ronda(c.get("round", {}).get("displayName", ""), slam=(lvl == "G"))
                    if rd is None:
                        continue
                    comp = c.get("competitors", [])
                    if len(comp) != 2:
                        continue
                    gan = [x for x in comp if x.get("winner")]
                    per = [x for x in comp if not x.get("winner")]
                    if len(gan) != 1 or len(per) != 1:
                        continue
                    try:
                        fecha = pd.Timestamp(c["date"]).tz_localize(None).normalize()
                    except Exception:
                        continue
                    if fecha < pd.Timestamp(desde):
                        continue   # fuera de la ventana; lo ya guardado lo protege el dedup por clave
                    w = resolver(gan[0]["athlete"]["displayName"])
                    l = resolver(per[0]["athlete"]["displayName"])
                    clave = (fecha, tname, rd, w, l)
                    if clave in vistos:
                        continue
                    vistos.add(clave)
                    bo = c.get("format", {}).get("regulation", {}).get("periods", 3) or 3
                    # marcador desde linescores del ganador (puede faltar)
                    ls_w = [x.get("value") for x in gan[0].get("linescores", [])]
                    ls_l = [x.get("value") for x in per[0].get("linescores", [])]
                    score = " ".join(f"{int(a)}-{int(b)}" for a, b in zip(ls_w, ls_l)) if ls_w and ls_l else ""
                    filas.append({
                        "tourney_date": int(fecha.strftime("%Y%m%d")), "tourney_name": tname,
                        "tourney_level": lvl, "surface": surf, "best_of": int(bo), "round": rd, "score": score,
                        "winner_name": w, "loser_name": l,
                        "winner_rank": rk.get(norm(w)), "loser_rank": rk.get(norm(l)),
                        "winner_age": edad(w, fecha), "loser_age": edad(l, fecha), "fecha": fecha,
                    })
        d += timedelta(days=3)
        time.sleep(0.05)

    if not filas:
        print("No hay partidos nuevos en ESPN.")
        return {"nuevos": 0, "hasta": corte}
    nuevo = pd.DataFrame(filas)
    for c in cols:
        if c not in nuevo.columns:
            nuevo[c] = np.nan          # stats de saque vacías
    nuevo = nuevo[cols]
    full = pd.concat([base, nuevo], ignore_index=True)
    full = full.drop_duplicates(subset=["fecha", "tourney_name", "round", "winner_name", "loser_name"], keep="first")
    # canonicalizar variantes de un mismo jugador (acentos/guiones entre fuentes) a la grafía más usada
    from collections import Counter
    freq = Counter(pd.concat([full.winner_name, full.loser_name]).dropna())
    por_norm = {}
    for nm in freq:
        por_norm.setdefault(norm(nm), []).append(nm)
    cmap = {v: max(vs, key=lambda x: freq[x]) for vs in por_norm.values() if len(vs) > 1 for v in vs}
    if cmap:
        full["winner_name"] = full["winner_name"].map(lambda x: cmap.get(x, x))
        full["loser_name"] = full["loser_name"].map(lambda x: cmap.get(x, x))
        print(f"  nombres unificados (variantes acento/guion): {len(cmap)} -> {sorted(set(cmap.values()))[:5]}")
    full = full.sort_values("fecha").reset_index(drop=True)
    full.to_csv(f"{DATA}/partidos.csv", index=False)
    print(f"\nAñadidos {len(nuevo)} partidos de ESPN. Total: {len(full)} | rango {full.fecha.min().date()} -> {full.fecha.max().date()}")
    print(f"  torneos ESPN nuevos: {sorted(nuevo.tourney_name.unique())[:15]}")
    print(f"  superficies nuevas: {nuevo.surface.value_counts().to_dict()}")
    nuevos_jug = sum(1 for n in set(nuevo.winner_name) | set(nuevo.loser_name)
                     if n not in set(base.winner_name) | set(base.loser_name))
    print(f"  jugadores que no estaban en el histórico: {nuevos_jug}")
    # netos = los que sobrevivieron al dedup (el re-barrido vuelve a traer lo ya guardado)
    return {"nuevos": len(full) - len(base), "hasta": full.fecha.max()}


if __name__ == "__main__":
    main()
