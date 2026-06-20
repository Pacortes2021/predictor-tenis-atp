"""Análisis riguroso del predictor de tenis: demuestra la fuga de datos del trabajo original,
optimiza el blend de Elo, hace selección de variables (VIF/forward/Lasso) con validación
TEMPORAL y compara modelos con métricas múltiples (log-loss, Brier, AUC, accuracy, calibración)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, accuracy_score
import sys; sys.path.insert(0, "/Users/pabloignaciocortesvielma/Downloads/Tenis_Predictor")
import motor as mo

M = mo.cargar()
D = M["D"]; C = mo.CANDIDATAS
print(f"Dataset modelable: {len(D)} partidos (burn-in n>=10) | {D.fecha.min().date()} -> {D.fecha.max().date()}")
tr, te = D[D.fecha < mo.CORTE_TEST], D[D.fecha >= mo.CORTE_TEST]
print(f"Train (<2023): {len(tr)} | Test (2023-2024): {len(te)}")
cv = TimeSeriesSplit(5)

def metrics(P, y):
    return dict(logloss=log_loss(y, P), brier=brier_score_loss(y, P),
                auc=roc_auc_score(y, P), acc=accuracy_score(y, (P >= .5).astype(int)))

# ============ (0) DEMOSTRACIÓN DE LA FUGA DEL TRABAJO ORIGINAL ============
from sklearn.ensemble import RandomForestClassifier
print("\n" + "="*70)
print("(0) FUGA DE DATOS")
print("="*70)
# reconstruir features orientadas al GANADOR (como en el original: ganador siempre lado +)
signo = (2*D.y - 1).values.reshape(-1, 1)
baseW = D[C].values * signo
Xd = np.vstack([baseW, -baseW]); yd = np.r_[np.ones(len(baseW)), np.zeros(len(baseW))]
gid = np.r_[np.arange(len(baseW)), np.arange(len(baseW))]   # par espejo comparte id
print("  (0a) Duplicar filas (+x y=1 / -x y=0) y luego KFold SHUFFLE (su método)")
print("       vs GroupKFold que mantiene cada par espejo junto:")
for nom, clf in [("Logística (su modelo)", LogisticRegression(max_iter=1000)),
                 ("RandomForest (modelo flexible)", RandomForestClassifier(n_estimators=60, max_depth=10, n_jobs=-1, random_state=0))]:
    p = Pipeline([("sc", StandardScaler()), ("m", clf)])
    a_sh = cross_val_score(p, Xd, yd, cv=KFold(5, shuffle=True, random_state=42), scoring="roc_auc").mean()
    a_gr = cross_val_score(p, Xd, yd, groups=gid, cv=GroupKFold(5), scoring="roc_auc").mean()
    print(f"     {nom:32} shuffle={a_sh:.4f}  group={a_gr:.4f}  fuga=+{a_sh-a_gr:.4f}")
print("  (0b) Sobre el dataset honesto (1 fila/partido), barajar el tiempo vs respetarlo:")
Xo, yo = D[mo.FEATS], D.y
ph = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(fit_intercept=False, max_iter=2000))])
a_shuf = cross_val_score(ph, Xo, yo, cv=KFold(5, shuffle=True, random_state=1), scoring="roc_auc").mean()
a_temp = cross_val_score(ph, Xo, yo, cv=TimeSeriesSplit(5), scoring="roc_auc").mean()
print(f"     KFold shuffle (mira el futuro) AUC={a_shuf:.4f}  vs  TimeSeriesSplit (temporal) AUC={a_temp:.4f}")

# ============ (1) OPTIMIZAR BLEND DE ELO (overall vs superficie) ============
print("\n" + "="*70)
print("(1) ¿Cuánto pesar Elo de superficie vs Elo general? (validación temporal)")
print("="*70)
def elo_blend_ll(w):
    Z = tr.copy(); Zt = te.copy()
    Z["b"] = (1-w)*Z.elo_overall_diff + w*Z.elo_surf_diff
    Zt["b"] = (1-w)*Zt.elo_overall_diff + w*Zt.elo_surf_diff
    m = Pipeline([("sc",StandardScaler()),("m",LogisticRegression(fit_intercept=False,max_iter=2000))]).fit(Z[["b"]], Z.y)
    return log_loss(Zt.y, m.predict_proba(Zt[["b"]])[:,1])
for w in [0.0,0.25,0.4,0.5,0.6,0.75,1.0]:
    print(f"  w_superficie={w:.2f}  ->  test log-loss={elo_blend_ll(w):.4f}")

# ============ (2) VIF ============
print("\n" + "="*70); print("(2) VIF (multicolinealidad sobre las candidatas)"); print("="*70)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
Xv = sm.add_constant(tr[C])
vif = pd.DataFrame({"feature": C, "VIF": [variance_inflation_factor(Xv.values, i+1) for i in range(len(C))]})
print(vif.sort_values("VIF", ascending=False).to_string(index=False))

# ============ (3) FORWARD (CV temporal, log-loss) ============
print("\n" + "="*70); print("(3) Selección forward (CV temporal por log-loss)"); print("="*70)
def ll_cv(cols):
    p = Pipeline([("sc",StandardScaler()),("m",LogisticRegression(fit_intercept=False,max_iter=2000))])
    return -cross_val_score(p, tr[cols], tr.y, cv=cv, scoring="neg_log_loss").mean()
sel, rem, best = [], C[:], 99
while rem:
    sc = {f: ll_cv(sel+[f]) for f in rem}; bf = min(sc, key=sc.get)
    if sc[bf] < best - 0.0005:
        sel.append(bf); rem.remove(bf); print(f"  + {bf:20} log-loss={sc[bf]:.4f}"); best = sc[bf]
    else:
        print(f"  (siguiente {bf} {sc[bf]:.4f}: no mejora -> paro)"); break
print(f"  SET FORWARD: {sel}")

# ============ (4) LASSO sobre todas ============
print("\n" + "="*70); print("(4) Lasso (L1) sobre TODAS las candidatas: qué sobrevive"); print("="*70)
for Cf in [0.02,0.05,0.1,0.3]:
    m = Pipeline([("sc",StandardScaler()),("m",LogisticRegression(penalty="l1",solver="saga",C=Cf,fit_intercept=False,max_iter=5000))]).fit(tr[C], tr.y)
    vivas = [f for k,f in enumerate(C) if abs(m.named_steps["m"].coef_[0,k])>1e-6]
    print(f"  C={Cf}: {len(vivas)} vivas -> {vivas}")

# ============ (5) COMPARACIÓN DE MODELOS EN TEST TEMPORAL ============
print("\n" + "="*70); print("(5) Comparación en TEST temporal (2023-2024) — múltiples métricas"); print("="*70)
print(f"  {'modelo':32} {'logloss':>8} {'brier':>7} {'auc':>7} {'acc':>7}")
print("  " + "-"*64)
modelos = {
    "Producción (eloOv+eloSurf+form)": mo.FEATS,
    "Solo elo_overall": ["elo_overall_diff"],
    "Solo elo_surf": ["elo_surf_diff"],
    "Elo overall+surf": ["elo_overall_diff","elo_surf_diff"],
    "log_rank (~ranking original)": ["log_rank_diff"],
    "Set forward": sel,
    "TODAS las candidatas": C,
}
for nom, cols in modelos.items():
    m = Pipeline([("sc",StandardScaler()),("m",LogisticRegression(fit_intercept=False,max_iter=2000))]).fit(tr[cols], tr.y)
    P = m.predict_proba(te[cols])[:,1]; mt = metrics(P, te.y)
    print(f"  {nom:32} {mt['logloss']:>8.4f} {mt['brier']:>7.4f} {mt['auc']:>7.4f} {mt['acc']:>7.3f}")
p0 = np.full(len(te), tr.y.mean())
print(f"  {'baseline (0.5)':32} {log_loss(te.y,np.c_[1-p0,p0]):>8.4f} {brier_score_loss(te.y,p0):>7.4f} {'-':>7} {'-':>7}")

# ============ (6) CALIBRACIÓN del modelo de producción ============
print("\n" + "="*70); print("(6) Calibración del modelo de producción (test)"); print("="*70)
m = Pipeline([("sc",StandardScaler()),("m",LogisticRegression(fit_intercept=False,max_iter=2000))]).fit(tr[mo.FEATS], tr.y)
P = m.predict_proba(te[mo.FEATS])[:,1]
te2 = te.copy(); te2["p"] = P
bins = pd.cut(te2.p, [0,.2,.4,.6,.8,1.0])
cal = te2.groupby(bins).agg(pred=("p","mean"), real=("y","mean"), n=("y","size"))
print(cal.to_string())
