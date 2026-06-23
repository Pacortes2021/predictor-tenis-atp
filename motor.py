"""Motor del predictor de tenis ATP (todas las superficies).

Aplica todo lo aprendido en los proyectos de fútbol, corrigiendo los errores del trabajo
original de Wimbledon:
  - Elo POR SUPERFICIE (Hard/Clay/Grass) con K decreciente (método Sackmann/538), en vez de rank_diff.
  - Features point-in-time (walk-forward, sin fuga de futuro).
  - Una fila por partido con orientación aleatoria + sin intercepto (antisimetría f(A,B)=-f(B,A)),
    en vez de duplicar filas (que con KFold shuffle filtraba el espejo a test).
  - Validación TEMPORAL (pasado->futuro), no KFold aleatorio.
  - Probabilidades calibradas y métricas múltiples (log-loss, Brier, AUC, accuracy, calibración).
  - Simulación Monte Carlo del cuadro (no determinista).
"""
import warnings
warnings.filterwarnings("ignore")
import os
from collections import defaultdict, deque
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ELO_INIT = 1500.0
SUPERFICIES = ["Hard", "Clay", "Grass"]
FORM_N = 20          # ventana de "forma reciente" (últimos N partidos)
CORTE_TEST = "2023-01-01"   # validación temporal: train < corte, test >= corte

# Features candidatas (todas son diferencias jugador1 - jugador2, point-in-time)
CANDIDATAS = [
    "elo_overall_diff", "elo_surf_diff", "log_rank_diff", "age_diff", "h2h_diff",
    "ace_rate_diff", "df_rate_diff", "firstIn_diff", "firstWon_diff", "secondWon_diff",
    "bpSaved_diff", "retWon_diff", "bpConv_diff", "form_diff",
]
# Modelo de producción: elegido por selección forward (CV temporal, log-loss). Ver analisis.py.
# Elo general + Elo de superficie + edad + log-ranking. Las stats de saque/resto NO aportan
# por encima de esto (todas vivas en Lasso pero +0.001 log-loss = ruido), así que se omiten.
FEATS = ["elo_overall_diff", "elo_surf_diff", "age_diff", "log_rank_diff"]


def _k(n):
    """K-factor decreciente: jugadores nuevos se mueven rápido, veteranos son estables (Sackmann)."""
    return 250.0 / (n + 5.0) ** 0.4


def _esperado(ra, rb):
    return 1.0 / (1.0 + 10.0 ** (-(ra - rb) / 400.0))


def _rate(num, den):
    return num / den if den > 0 else np.nan


def _construir(df):
    """Recorre los partidos en orden cronológico calculando Elo y features ANTES de cada partido.
    Devuelve el dataset de entrenamiento, los ratings/estado final por jugador, y el h2h."""
    elo_ov = defaultdict(lambda: ELO_INIT)
    elo_su = {s: defaultdict(lambda: ELO_INIT) for s in SUPERFICIES}
    n_ov = defaultdict(int)
    n_su = {s: defaultdict(int) for s in SUPERFICIES}
    car = defaultdict(lambda: defaultdict(float))   # acumulados de carrera (saque/resto)
    form = defaultdict(lambda: deque(maxlen=FORM_N))
    h2h = defaultdict(lambda: [0, 0])
    estado = {}  # estado final por jugador (para predecir partidos nuevos)
    filas = []

    def perfil(j, s):
        c = car[j]
        return {
            "elo_ov": elo_ov[j], "elo_su": elo_su[s][j],
            "ace": _rate(c["ace"], c["svpt"]), "df": _rate(c["df"], c["svpt"]),
            "firstIn": _rate(c["1stIn"], c["svpt"]), "firstWon": _rate(c["1stWon"], c["1stIn"]),
            "secondWon": _rate(c["2ndWon"], c["svpt"] - c["1stIn"]),
            "bpSaved": _rate(c["bpSaved"], c["bpFaced"]),
            "retWon": _rate(c["retWon"], c["retFaced"]), "bpConv": _rate(c["bpConv"], c["bpOpp"]),
            "form": (np.mean(form[j]) if form[j] else np.nan),
            "rank": None,
        }

    for r in df.itertuples(index=False):
        w, l, s = r.winner_name, r.loser_name, r.surface
        pw, pl = perfil(w, s), perfil(l, s)
        pw["rank"], pl["rank"] = r.winner_rank, r.loser_rank
        hk = tuple(sorted((w, l)))
        h2h_w = h2h[hk][0] - h2h[hk][1] if hk[0] == w else h2h[hk][1] - h2h[hk][0]

        # fila con orientación: ganador como "j1" (outcome=1). La aleatorización se hace después.
        def diffs(a, b, age_a, age_b, h2h_a):
            lr = lambda rk: -np.log(rk) if (rk and rk > 0) else -np.log(1500)
            return {
                "elo_overall_diff": a["elo_ov"] - b["elo_ov"],
                "elo_surf_diff": a["elo_su"] - b["elo_su"],
                "log_rank_diff": lr(a["rank"]) - lr(b["rank"]),
                "age_diff": (age_a or 0) - (age_b or 0),
                "h2h_diff": h2h_a,
                "ace_rate_diff": a["ace"] - b["ace"], "df_rate_diff": a["df"] - b["df"],
                "firstIn_diff": a["firstIn"] - b["firstIn"], "firstWon_diff": a["firstWon"] - b["firstWon"],
                "secondWon_diff": a["secondWon"] - b["secondWon"], "bpSaved_diff": a["bpSaved"] - b["bpSaved"],
                "retWon_diff": a["retWon"] - b["retWon"], "bpConv_diff": a["bpConv"] - b["bpConv"],
                "form_diff": (a["form"] if not np.isnan(a["form"]) else 0.5) - (b["form"] if not np.isnan(b["form"]) else 0.5),
            }

        d = diffs(pw, pl, r.winner_age, r.loser_age, h2h_w)
        d.update({"fecha": r.fecha, "surface": s, "best_of": r.best_of,
                  "n_min": min(n_ov[w], n_ov[l])})  # nº de partidos del menos experimentado
        filas.append(d)

        # ---- actualizar Elo (overall + superficie) ----
        ea = _esperado(elo_ov[w], elo_ov[l])
        kw, kl = _k(n_ov[w]), _k(n_ov[l])
        elo_ov[w] += kw * (1 - ea); elo_ov[l] += kl * (0 - (1 - ea))
        esa = _esperado(elo_su[s][w], elo_su[s][l])
        ksw, ksl = _k(n_su[s][w]), _k(n_su[s][l])
        elo_su[s][w] += ksw * (1 - esa); elo_su[s][l] += ksl * (0 - (1 - esa))
        n_ov[w] += 1; n_ov[l] += 1; n_su[s][w] += 1; n_su[s][l] += 1
        # ---- h2h, forma ----
        if hk[0] == w: h2h[hk][0] += 1
        else: h2h[hk][1] += 1
        form[w].append(1); form[l].append(0)
        # ---- acumulados de saque/resto (point-in-time) ----
        def acum(j, pre, opp):
            c = car[j]
            for st in ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "bpSaved", "bpFaced"]:
                v = getattr(r, f"{pre}_{st}", np.nan)
                if not pd.isna(v): c[st] += v
            # resto: puntos ganados al resto = svpt_rival - (1stWon+2ndWon)_rival
            osv, o1, o2 = getattr(r, f"{opp}_svpt", np.nan), getattr(r, f"{opp}_1stWon", np.nan), getattr(r, f"{opp}_2ndWon", np.nan)
            if not (pd.isna(osv) or pd.isna(o1) or pd.isna(o2)):
                c["retWon"] += osv - (o1 + o2); c["retFaced"] += osv
            obpf, obps = getattr(r, f"{opp}_bpFaced", np.nan), getattr(r, f"{opp}_bpSaved", np.nan)
            if not (pd.isna(obpf) or pd.isna(obps)):
                c["bpConv"] += obpf - obps; c["bpOpp"] += obpf
        acum(w, "w", "l"); acum(l, "l", "w")
        # ---- guardar estado final (último visto) por jugador ----
        for j, age, rk in [(w, r.winner_age, r.winner_rank), (l, r.loser_age, r.loser_rank)]:
            estado[j] = {"elo_ov": elo_ov[j], "n": n_ov[j], "age": age, "rank": rk, "ultima": r.fecha,
                         "elo_su": {s2: elo_su[s2][j] for s2 in SUPERFICIES},
                         "n_su": {s2: n_su[s2][j] for s2 in SUPERFICIES},
                         **{k2: perfil(j, "Hard")[k2] for k2 in ["ace", "df", "firstIn", "firstWon",
                                                                  "secondWon", "bpSaved", "retWon", "bpConv", "form"]}}
    D = pd.DataFrame(filas)
    return D, estado, h2h


def cargar():
    """Carga datos, construye el dataset point-in-time y entrena el modelo de producción."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(f"{DATA}/partidos.csv", parse_dates=["fecha"])
    df = df.dropna(subset=["winner_name", "loser_name", "surface", "fecha"]).reset_index(drop=True)
    D, estado, h2h = _construir(df)

    # quitar partidos sin historia suficiente (burn-in): ambos con >=10 partidos previos
    Dm = D[D["n_min"] >= 10].copy().reset_index(drop=True)
    for c in CANDIDATAS:
        Dm[c] = Dm[c].fillna(0.0)
    # orientación aleatoria: 1 fila por partido, mitad ganador como j1 (y=1), mitad como j2 (y=0)
    rng = np.random.RandomState(42)
    flip = rng.rand(len(Dm)) < 0.5
    X = Dm[CANDIDATAS].copy()
    X.loc[flip, :] = -X.loc[flip, :]
    y = (~flip).astype(int)   # si NO flip -> j1 es el ganador real -> y=1
    Dm["y"] = y
    for c in CANDIDATAS:
        Dm[c] = X[c].values

    # entrenar modelo de producción (Elo blend + forma), SIN intercepto (antisimetría)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(fit_intercept=False, max_iter=2000))])
    pipe.fit(Dm[FEATS], Dm["y"])

    return {"df": df, "D": Dm, "estado": estado, "h2h": h2h, "pipe": pipe, "feats": FEATS}


# ----------------------------- predicción de un partido -----------------------------
def _perfil_pred(estado, j, surface):
    """Estado actual de un jugador para predecir; jugador desconocido -> perfil neutro."""
    if j in estado:
        e = estado[j]
        return {"elo_ov": e["elo_ov"], "elo_su": e["elo_su"].get(surface, ELO_INIT),
                "n_su": e["n_su"].get(surface, 0), "rank": e["rank"], "age": e["age"],
                "form": e["form"] if not (e["form"] is None or (isinstance(e["form"], float) and np.isnan(e["form"]))) else 0.5,
                "conocido": True}
    return {"elo_ov": ELO_INIT, "elo_su": ELO_INIT, "n_su": 0, "rank": 1500, "age": 24,
            "form": 0.5, "conocido": False}


def prob_partido(M, j1, j2, surface):
    """Probabilidad de que j1 le gane a j2 en la superficie dada."""
    a = _perfil_pred(M["estado"], j1, surface)
    b = _perfil_pred(M["estado"], j2, surface)
    hk = tuple(sorted((j1, j2)))
    h = M["h2h"].get(hk, [0, 0]); h2h_a = (h[0] - h[1]) if hk[0] == j1 else (h[1] - h[0])
    feat = {
        "elo_overall_diff": a["elo_ov"] - b["elo_ov"],
        "elo_surf_diff": a["elo_su"] - b["elo_su"],
        "form_diff": a["form"] - b["form"],
        "age_diff": (a["age"] or 24) - (b["age"] or 24),
        "log_rank_diff": (-np.log(a["rank"] or 1500)) - (-np.log(b["rank"] or 1500)),
        "h2h_diff": h2h_a,
    }
    x = pd.DataFrame([{c: feat.get(c, 0.0) for c in M["feats"]}])
    p = float(M["pipe"].predict_proba(x)[0, 1])
    return p, a, b


def jugadores_activos(M, desde="2023-01-01", min_partidos=15):
    """Lista de jugadores con actividad reciente, ordenada por Elo general (para los selectores)."""
    desde = pd.Timestamp(desde)
    js = [(j, e["elo_ov"]) for j, e in M["estado"].items()
          if e.get("ultima") is not None and e["ultima"] >= desde and e["n"] >= min_partidos]
    return [j for j, _ in sorted(js, key=lambda x: -x[1])]


def _invertir_set(p_match, bo):
    """Dado P(ganar el partido), halla P(ganar un set) asumiendo sets i.i.d. (bisección)."""
    f = lambda p: (p * p * (3 - 2 * p) if bo == 3 else p ** 3 * (10 - 15 * p + 6 * p ** 2)) - p_match
    lo, hi = 1e-9, 1 - 1e-9
    for _ in range(64):
        m = (lo + hi) / 2
        if f(m) > 0: hi = m
        else: lo = m
    return (lo + hi) / 2


def distribucion_sets(p_match, bo=3):
    """Distribución del marcador en sets (desde la perspectiva del jugador con prob p_match).
    Devuelve (dict marcador->prob, p_set). Análogo a la matriz de marcadores del fútbol."""
    p = _invertir_set(p_match, bo); q = 1 - p
    if bo == 3:
        d = {"2-0": p * p, "2-1": 2 * p * p * q, "1-2": 2 * p * q * q, "0-2": q * q}
    else:
        d = {"3-0": p ** 3, "3-1": 3 * p ** 3 * q, "3-2": 6 * p ** 3 * q * q,
             "2-3": 6 * p * p * q ** 3, "1-3": 3 * p * q ** 3, "0-3": q ** 3}
    return d, p


def _race(p, n):
    """P(ganar una 'carrera' a n puntos, ganando por 2, con puntos i.i.d. de prob p) — game (n=4) o tiebreak (n=7)."""
    from math import comb
    pre = sum(comb(n - 1 + k, k) * p ** n * (1 - p) ** k for k in range(n - 1))
    deuce = comb(2 * (n - 1), n - 1) * p ** (n - 1) * (1 - p) ** (n - 1)
    return pre + deuce * p ** 2 / (p ** 2 + (1 - p) ** 2)


def _p_set_de_punto(q):
    """P(ganar un set) a partir de la prob de ganar un punto q (modelo i.i.d. simplificado: game→set con tiebreak)."""
    from math import comb
    g = _race(q, 4); t = _race(q, 7)
    win_6k = sum(comb(5 + k, k) * g ** 6 * (1 - g) ** k for k in range(5))   # 6-0..6-4
    p55 = comb(10, 5) * g ** 5 * (1 - g) ** 5
    return win_6k + p55 * g * g + p55 * 2 * g * (1 - g) * t                   # +7-5 +tiebreak


def puntos_implicitos(p_match, bo=3):
    """% de puntos que gana (en promedio) el favorito implícito por su prob de partido.
    Revela lo fino del margen: un 'favorito 90%' suele ganar solo ~57% de los puntos."""
    def pm(q):
        ps = _p_set_de_punto(q)
        return ps * ps * (3 - 2 * ps) if bo == 3 else ps ** 3 * (10 - 15 * ps + 6 * ps ** 2)
    lo, hi = 1e-4, 1 - 1e-4
    for _ in range(64):
        m = (lo + hi) / 2
        if pm(m) > p_match: hi = m
        else: lo = m
    return (lo + hi) / 2


def ranking_elo(M, surface=None, top=30, min_partidos=20):
    """Ranking de jugadores por Elo (overall o de una superficie)."""
    filas = []
    for j, e in M["estado"].items():
        if surface:
            if e["n_su"].get(surface, 0) < min_partidos:
                continue
            elo = e["elo_su"].get(surface, ELO_INIT)
        else:
            if e["n"] < min_partidos:
                continue
            elo = e["elo_ov"]
        filas.append({"jugador": j, "elo": round(elo), "partidos": e["n"]})
    return pd.DataFrame(filas).sort_values("elo", ascending=False).head(top).reset_index(drop=True)


def ultimos_partidos(M, jugador, surface=None, n=8):
    """Últimos n partidos de un jugador (opcionalmente solo de una superficie)."""
    df = M["df"]
    m = df[(df["winner_name"] == jugador) | (df["loser_name"] == jugador)]
    if surface:
        m = m[m["surface"] == surface]
    m = m.sort_values("fecha", ascending=False).head(n)
    filas = []
    for r in m.itertuples(index=False):
        gano = r.winner_name == jugador
        filas.append({
            "Fecha": r.fecha.date(), "Res": "✅" if gano else "❌",
            "Rival": r.loser_name if gano else r.winner_name,
            "Marcador": (r.score if isinstance(r.score, str) and r.score else "—"),
            "Sup.": r.surface, "Torneo": r.tourney_name,
        })
    return pd.DataFrame(filas)


def historial_versus(M, j1, j2, surface=None):
    """Todos los enfrentamientos directos entre j1 y j2 (opcionalmente por superficie)."""
    df = M["df"]
    m = df[((df["winner_name"] == j1) & (df["loser_name"] == j2)) |
           ((df["winner_name"] == j2) & (df["loser_name"] == j1))]
    if surface:
        m = m[m["surface"] == surface]
    m = m.sort_values("fecha", ascending=False)
    filas = []
    for r in m.itertuples(index=False):
        filas.append({
            "Fecha": r.fecha.date(), "Ganador": r.winner_name,
            "Marcador": (r.score if isinstance(r.score, str) and r.score else "—"),
            "Sup.": r.surface, "Torneo": r.tourney_name,
        })
    return pd.DataFrame(filas)


# ----------------------------- simulación Monte Carlo del cuadro -----------------------------
def simular_torneo(M, draw, surface, n_sims=10000, seed=0):
    """draw: lista de nombres (potencia de 2, en orden del cuadro). Devuelve P(campeón) por jugador.
    A diferencia del original (avanzar siempre al de p>=0.5), aquí se sortea cada partido."""
    rng = np.random.RandomState(seed)
    n = len(draw)
    # precalcular matriz de probabilidades p[i][j] = P(i gana a j)
    P = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                P[(i, j)] = prob_partido(M, draw[i], draw[j], surface)[0]
    titulos = np.zeros(n)
    finales = np.zeros(n)
    for _ in range(n_sims):
        vivos = list(range(n))
        ronda = 0
        while len(vivos) > 1:
            nxt = []
            for k in range(0, len(vivos), 2):
                a, b = vivos[k], vivos[k + 1]
                ganador = a if rng.rand() < P[(a, b)] else b
                nxt.append(ganador)
            if len(nxt) == 1:
                finales[vivos[0]] += 1; finales[vivos[1]] += 1
            vivos = nxt
        titulos[vivos[0]] += 1
    res = pd.DataFrame({"jugador": draw,
                        "P_campeon": titulos / n_sims,
                        "P_final": finales / n_sims})
    return res.sort_values("P_campeon", ascending=False).reset_index(drop=True)
