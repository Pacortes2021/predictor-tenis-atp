"""Dos preguntas del usuario, respondidas con datos:
  A) ¿Hacían falta datos desde 2000, o basta con menos historia?
  B) ¿Otros clasificadores baten a la regresión logística?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import sys; sys.path.insert(0, "/Users/pabloignaciocortesvielma/Downloads/Tenis_Predictor")
import motor as mo

df_full = pd.read_csv(f"{mo.DATA}/partidos.csv", parse_dates=["fecha"]).dropna(
    subset=["winner_name", "loser_name", "surface", "fecha"]).reset_index(drop=True)


def dataset_desde(anio):
    """Reconstruye Elo y features usando solo datos >= anio, devuelve dataset orientado."""
    df = df_full[df_full.fecha >= f"{anio}-01-01"].reset_index(drop=True)
    D, _, _ = mo._construir(df)
    D = D[D["n_min"] >= 10].copy().reset_index(drop=True)
    for c in mo.CANDIDATAS:
        D[c] = D[c].fillna(0.0)
    rng = np.random.RandomState(42); flip = rng.rand(len(D)) < 0.5
    X = D[mo.CANDIDATAS].copy(); X.loc[flip, :] = -X.loc[flip, :]
    D["y"] = (~flip).astype(int)
    for c in mo.CANDIDATAS:
        D[c] = X[c].values
    return D


print("="*72)
print("(A) ¿Cuánta historia hace falta? (rebuild completo desde cada año; test 2023-2024)")
print("="*72)
print(f"  {'desde':>6} {'train n':>9} {'logloss':>9} {'auc':>7} {'acc':>7}")
for anio in [2000, 2008, 2014, 2018, 2020, 2021]:
    D = dataset_desde(anio)
    tr, te = D[D.fecha < "2023-01-01"], D[D.fecha >= "2023-01-01"]
    if len(te) < 1000:
        print(f"  {anio:>6}  test insuficiente"); continue
    m = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(fit_intercept=False, max_iter=2000))]).fit(tr[mo.FEATS], tr.y)
    P = m.predict_proba(te[mo.FEATS])[:, 1]
    print(f"  {anio:>6} {len(tr):>9} {log_loss(te.y,P):>9.4f} {roc_auc_score(te.y,P):>7.4f} {accuracy_score(te.y,(P>=.5).astype(int)):>7.3f}")

print("\n" + "="*72)
print("(B) ¿Otros clasificadores? (mismas FEATS, train<2023, test 2023-2024)")
print("="*72)
D = dataset_desde(2000)
tr, te = D[D.fecha < "2023-01-01"], D[D.fecha >= "2023-01-01"]
Xtr, Xte, ytr, yte = tr[mo.FEATS], te[mo.FEATS], tr.y, te.y
clfs = {
    "Logística (producción)": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(fit_intercept=False, max_iter=2000))]),
    "Ridge (L2)": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(penalty="l2", C=1.0, fit_intercept=False, max_iter=2000))]),
    "Lasso (L1)": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(penalty="l1", solver="saga", C=0.5, fit_intercept=False, max_iter=5000))]),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=30, n_jobs=-1, random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=0),
}
print(f"  {'clasificador':26} {'logloss':>9} {'auc':>7} {'acc':>7}")
print("  " + "-"*52)
for nom, clf in clfs.items():
    clf.fit(Xtr, ytr)
    P = clf.predict_proba(Xte)[:, 1]
    print(f"  {nom:26} {log_loss(yte,P):>9.4f} {roc_auc_score(yte,P):>7.4f} {accuracy_score(yte,(P>=.5).astype(int)):>7.3f}")
