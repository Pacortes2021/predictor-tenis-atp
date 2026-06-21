"""Panel Streamlit: predictor de tenis ATP en todas las superficies (Hard/Clay/Grass).
Reconstrucción rigurosa del trabajo original de Wimbledon con Elo por superficie,
validación temporal, métricas múltiples y simulación Monte Carlo."""
import numpy as np
import pandas as pd
import streamlit as st
import motor as mo

st.set_page_config(page_title="Predictor de Tenis ATP", page_icon="🎾", layout="wide")

# --------------------------- estilo ---------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(160deg,#0a3d2c 0%,#10261d 100%); }
h1,h2,h3,h4,p,label,span,div { color:#eafff3; }
.bloque { background:rgba(255,255,255,.05); border:1px solid rgba(204,255,0,.25);
          border-radius:14px; padding:16px 20px; margin-bottom:14px; }
.gana { color:#ccff00; font-weight:800; }
.mut { color:#9fc7b4; font-size:.86rem; }
[data-testid="stMetricValue"] { color:#ccff00; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:rgba(255,255,255,.05); border-radius:10px 10px 0 0; padding:8px 16px; }
.stTabs [aria-selected="true"] { background:rgba(204,255,0,.18); }
</style>
""", unsafe_allow_html=True)

SURF = {"Dura 🟦": "Hard", "Tierra 🟧": "Clay", "Pasto 🟩": "Grass"}


@st.cache_resource(show_spinner="Calculando Elo de 60.000+ partidos...")
def cargar():
    return mo.cargar()


M = cargar()
activos = mo.jugadores_activos(M)
ULT = M["df"]["fecha"].max()

st.title("🎾 Predictor de Tenis ATP")
st.markdown(f"<p class='mut'>Elo por superficie + validación temporal + Monte Carlo · datos ATP 2000 → {ULT.strftime('%d-%b-%Y')} "
            f"(histórico Sackmann + ESPN en vivo, todas las superficies)</p>",
            unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["🎾 Predecir partido", "📊 Rankings Elo", "🏆 Simulador de torneo", "🎯 El modelo"])

# ============================ TAB 1: PREDECIR ============================
with t1:
    c1, c2, c3 = st.columns([3, 3, 2])
    with c1:
        j1 = st.selectbox("Jugador 1", activos, index=activos.index("Carlos Alcaraz") if "Carlos Alcaraz" in activos else 0)
    with c2:
        j2 = st.selectbox("Jugador 2", activos, index=activos.index("Jannik Sinner") if "Jannik Sinner" in activos else 1)
    with c3:
        sup_lbl = st.radio("Superficie", list(SURF.keys()), horizontal=False)
    sup = SURF[sup_lbl]

    if j1 == j2:
        st.warning("Elige dos jugadores distintos.")
    else:
        p, a, b = mo.prob_partido(M, j1, j2, sup)
        gan, pg = (j1, p) if p >= .5 else (j2, 1 - p)
        st.markdown(f"<div class='bloque'><h3>Favorito en {sup_lbl}: <span class='gana'>{gan}</span> "
                    f"&nbsp;·&nbsp; {pg:.1%}</h3></div>", unsafe_allow_html=True)

        cc = st.columns(2)
        for col, nom, prob, est in [(cc[0], j1, p, a), (cc[1], j2, 1 - p, b)]:
            with col:
                cuota = 1 / prob if prob > 0 else 99
                st.markdown(f"<div class='bloque'><h4>{nom}</h4>"
                            f"<p style='font-size:2rem;margin:.2rem 0' class='gana'>{prob:.1%}</p>"
                            f"<p class='mut'>Cuota justa: <b>{cuota:.2f}</b></p>"
                            f"<p class='mut'>Elo general: <b>{est['elo_ov']:.0f}</b> · "
                            f"Elo {sup_lbl.split()[0].lower()}: <b>{est['elo_su']:.0f}</b><br>"
                            f"Forma reciente: <b>{(est['form']*100):.0f}%</b> · "
                            f"Ranking ATP: <b>{int(est['rank']) if est['rank'] and not np.isnan(est['rank']) else '—'}</b></p></div>",
                            unsafe_allow_html=True)

        # barra de probabilidad
        st.markdown(f"""<div style="display:flex;height:34px;border-radius:8px;overflow:hidden;margin-top:4px">
        <div style="width:{p*100:.1f}%;background:#ccff00;color:#0a3d2c;display:flex;align-items:center;
        justify-content:center;font-weight:800">{p:.0%}</div>
        <div style="width:{(1-p)*100:.1f}%;background:#356b54;display:flex;align-items:center;
        justify-content:center;font-weight:700">{1-p:.0%}</div></div>
        <p class='mut' style='display:flex;justify-content:space-between'><span>{j1}</span><span>{j2}</span></p>""",
                    unsafe_allow_html=True)

        st.caption("La probabilidad combina Elo general + Elo de la superficie + edad + ranking ATP (modelo elegido por selección forward).")
        st.caption("⚠️ El **Elo** es una medida propia de fuerza (no existe un Elo oficial) y **no es lo mismo que el ranking ATP**. "
                   "El 'Ranking ATP' mostrado proviene del ranking actual de ESPN.")

        # ---- filtro de superficie para el historial ----
        st.markdown("---")
        filtro = st.radio("Historial a mostrar:", ["Todas las superficies", f"Solo {sup_lbl}"],
                          horizontal=True, key="filtro_hist")
        surf_f = None if filtro.startswith("Todas") else sup

        # ---- últimos partidos de cada jugador ----
        st.markdown("#### 📋 Últimos partidos")
        colp = st.columns(2)
        for col, jug in [(colp[0], j1), (colp[1], j2)]:
            with col:
                st.markdown(f"**{jug}**")
                up = mo.ultimos_partidos(M, jug, surf_f, n=8)
                if up.empty:
                    st.caption("Sin partidos para este filtro.")
                else:
                    st.dataframe(up, hide_index=True, width="stretch")

        # ---- head-to-head detallado ----
        st.markdown("#### 🆚 Enfrentamientos directos (head-to-head)")
        hv = mo.historial_versus(M, j1, j2, surf_f)
        if hv.empty:
            extra = "" if surf_f is None else f" en {sup_lbl}"
            st.info(f"No hay enfrentamientos directos{extra} entre **{j1}** y **{j2}** en los datos (desde 2000).")
        else:
            g1 = int((hv["Ganador"] == j1).sum())
            g2 = int((hv["Ganador"] == j2).sum())
            lider = j1 if g1 > g2 else (j2 if g2 > g1 else "empate")
            st.markdown(f"**{j1} {g1}–{g2} {j2}**  ·  {len(hv)} enfrentamiento{'s' if len(hv) != 1 else ''}"
                        + (f"  ·  lidera **{lider}**" if lider != "empate" else "  ·  igualados"))
            st.dataframe(hv, hide_index=True, width="stretch")

# ============================ TAB 2: RANKINGS ============================
with t2:
    cs = st.columns([2, 2, 3])
    with cs[0]:
        vista = st.selectbox("Ranking Elo por", ["General", "Dura (Hard)", "Tierra (Clay)", "Pasto (Grass)"])
    with cs[1]:
        solo_act = st.checkbox("Solo jugadores activos (2023+)", value=True)
    surf = {"General": None, "Dura (Hard)": "Hard", "Tierra (Clay)": "Clay", "Pasto (Grass)": "Grass"}[vista]
    rk = mo.ranking_elo(M, surface=surf, top=200, min_partidos=20)
    if solo_act:
        act = set(activos)
        rk = rk[rk.jugador.isin(act)]
    rk = rk.head(25).reset_index(drop=True)
    rk.index = rk.index + 1
    rk.columns = ["Jugador", "Elo", "Partidos (carrera)"]
    st.dataframe(rk, width='stretch', height=560)
    st.caption("El Elo de superficie revela especialistas: Nadal domina en tierra, Djokovic en pasto. "
               "Un jugador retirado conserva su último Elo (por eso el filtro 'activos').")
    st.info("Este ranking Elo **no es el ranking ATP** y no tiene por qué coincidir. El ATP suma puntos de los "
            "torneos de las últimas 52 semanas (premia jugar y ganar títulos, los puntos caducan). El Elo mide "
            "fuerza predictiva ajustada por rival y no caduca por inactividad. Ej.: hoy el Elo pone a Sinner #1, "
            "pero el ATP a Alcaraz #1 — su suspensión de 2025 le restó puntos ATP, no tanto Elo.")

# ============================ TAB 3: SIMULADOR ============================
with t3:
    st.markdown("#### Simulación Monte Carlo de un cuadro")
    st.caption("Elige los participantes (potencia de 2: 4, 8 o 16) y la superficie. Se sortea cada partido "
               "según su probabilidad y se repite 5.000 veces → probabilidad de ser campeón. "
               "**No** es determinista como el cuadro original (donde el favorito siempre avanzaba).")
    cc = st.columns([4, 2])
    with cc[0]:
        top16 = activos[:16]
        elegidos = st.multiselect("Participantes", activos, default=top16[:8])
    with cc[1]:
        sup2 = SURF[st.radio("Superficie del torneo", list(SURF.keys()), horizontal=False, key="sim")]
    n = len(elegidos)
    if n not in (4, 8, 16):
        st.warning(f"Elegiste {n}. Necesitas 4, 8 o 16 jugadores para un cuadro completo.")
    else:
        if st.button("🎲 Simular torneo (5.000 veces)", type="primary"):
            with st.spinner("Simulando..."):
                res = mo.simular_torneo(M, elegidos, sup2, n_sims=5000)
            res2 = res.copy()
            res2["P_campeon"] = (res2["P_campeon"] * 100).round(1).astype(str) + "%"
            res2["P_final"] = (res2["P_final"] * 100).round(1).astype(str) + "%"
            res2.columns = ["Jugador", "P(campeón)", "P(llegar a la final)"]
            res2.index = res2.index + 1
            st.dataframe(res2, width='stretch', height=min(560, 60 + 35 * n))
            campeon = res.iloc[0]
            st.success(f"🏆 Más probable campeón: **{campeon.jugador}** ({campeon.P_campeon:.1%})")

# ============================ TAB 4: EL MODELO ============================
with t4:
    st.markdown("### Cómo se hizo (y qué se corrigió del trabajo original)")
    st.markdown("""
Este predictor reconstruye un trabajo universitario previo (predicción de Wimbledon, solo pasto)
aplicando metodología rigurosa. Los cambios clave:
""")
    comp = pd.DataFrame({
        "Tema": ["Fuerza del jugador", "Superficies", "Validación", "Selección de modelo",
                 "Simulación del cuadro", "Ranking"],
        "Trabajo original": ["rank_diff (proxy débil)", "Solo pasto (grass_win_pct)",
                             "KFold aleatorio sobre filas duplicadas", "Por AUC (y por 0.0001)",
                             "Determinista (favorito siempre avanza)", "Ranking ATP crudo"],
        "Esta versión": ["Elo general + Elo POR superficie (K decreciente)", "Hard / Clay / Grass",
                         "Temporal (pasado→futuro), 1 fila/partido sin intercepto",
                         "Forward por log-loss + calibración (múltiples métricas)",
                         "Monte Carlo (5.000 sorteos → P(campeón))", "log-ranking + Elo"],
    })
    st.table(comp)

    st.markdown("### Rendimiento real (test temporal 2023–2024, 5.104 partidos)")
    perf = pd.DataFrame({
        "Modelo": ["Producción (Elo gen+sup+edad+rank)", "Solo Elo general", "Solo log-ranking (≈ original)",
                   "Con TODAS las stats (14 vars)", "Baseline (50/50)"],
        "log-loss": [0.6214, 0.6311, 0.6373, 0.6203, 0.6931],
        "Brier": [0.2167, 0.2207, 0.2237, 0.2163, 0.2500],
        "AUC": [0.7089, 0.6955, 0.6848, 0.7100, 0.500],
        "Accuracy": ["64.4%", "63.7%", "63.0%", "64.5%", "—"],
    })
    st.table(perf)
    st.markdown("""
- **El Elo manda**, igual que en fútbol: añadir Elo (general+superficie) sobre el ranking sube el AUC de 0.685 a 0.709 y el acierto de 63.0% a 64.4%.
- **Las stats de saque/resto casi no aportan**: meter las 14 variables solo mejora 0.001 en log-loss → se descartan por parsimonia (el Lasso las mantiene todas, pero no ayudan fuera de muestra).
- **El peso óptimo del Elo de superficie es ~0.4** (medido por validación temporal): la superficie importa, pero el Elo general aporta estabilidad.
""")

    st.markdown("### El hallazgo honesto sobre la 'fuga de datos'")
    st.info("""Esperaba que duplicar filas (+x/−x) y barajar con KFold inflara mucho el AUC original.
**Lo medí: el impacto real fue +0.0003 de AUC** (casi nada), porque un modelo lineal no puede
explotar el espejo. La metodología era incorrecta, pero el número no estaba muy inflado.
*Aun así* se corrigió: con un modelo flexible (RandomForest) o más variables, esa fuga sí muerde.
Medir antes de afirmar.""")

    st.markdown("### Calibración (test)")
    cal = pd.DataFrame({
        "Prob. predicha": ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
        "Predicho (medio)": ["14.1%", "30.8%", "50.0%", "69.1%", "86.3%"],
        "Real": ["13.2%", "34.5%", "50.0%", "64.8%", "83.3%"],
    })
    st.table(cal)
    st.caption("Bien calibrado en general; leve exceso de confianza en el tramo 60–80% (dice 69%, ocurre 65%).")
