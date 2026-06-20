"""Responde 3 preguntas con datos:
  1) ¿Es significativo el head-to-head (versus)? ¿Y la forma reciente (% ganados últimos X)?
  2) Barrido de ventanas de forma reciente (5/10/20/40 partidos).
  3) Modelo actual (con Elo) vs modelo full-stats SIN Elo."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import sys; sys.path.insert(0, "/Users/pabloignaciocortesvielma/Downloads/Tenis_Predictor")
import motor as mo

df_full = pd.read_csv(f"{mo.DATA}/partidos.csv", parse_dates=["fecha"]).dropna(
    subset=["winner_name", "loser_name", "surface", "fecha"]).reset_index(drop=True)


def dataset(form_n=20):
    mo.FORM_N = form_n
    D, _, _ = mo._construir(df_full)
    D = D[D["n_min"] >= 10].copy().reset_index(drop=True)
    for c in mo.CANDIDATAS:
        D[c] = D[c].fillna(0.0)
    rng = np.random.RandomState(42); flip = rng.rand(len(D)) < 0.5
    X = D[mo.CANDIDATAS].copy(); X.loc[flip, :] = -X.loc[flip, :]
    D["y"] = (~flip).astype(int)
    for c in mo.CANDIDATAS:
        D[c] = X[c].values
    return D


def evalua(tr, te, cols):
    m = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(fit_intercept=False, max_iter=3000))]).fit(tr[cols], tr.y)
    P = m.predict_proba(te[cols])[:, 1]
    return log_loss(te.y, P), roc_auc_score(te.y, P), accuracy_score(te.y, (P >= .5).astype(int))


D = dataset(20)
tr, te = D[D.fecha < "2023-01-01"], D[D.fecha >= "2023-01-01"]

# ---------- 1) p-values (Logit sobre TODAS las candidatas, estandarizadas, train) ----------
print("="*68); print("(1) Significancia: coeficientes y p-values (train, sin intercepto)"); print("="*68)
Xs = pd.DataFrame(StandardScaler().fit_transform(tr[mo.CANDIDATAS]), columns=mo.CANDIDATAS)
res = sm.Logit(tr.y.values, Xs).fit(disp=0)
tab = pd.DataFrame({"coef": res.params, "p_value": res.pvalues}).sort_values("p_value")
for f, row in tab.iterrows():
    sig = "***" if row.p_value < .001 else ("**" if row.p_value < .01 else ("*" if row.p_value < .05 else "  ns"))
    print(f"  {f:20} coef={row.coef:+.3f}  p={row.p_value:.4f} {sig}")

# ---------- 2) valor incremental de versus y forma sobre el modelo de producción ----------
print("\n" + "="*68); print("(2) ¿Aportan VERSUS y FORMA sobre el modelo de producción?"); print("="*68)
print(f"  {'modelo':34} {'logloss':>9} {'auc':>7} {'acc':>7}")
for nom, cols in [("Producción (4 vars)", mo.FEATS),
                  ("  + h2h_diff (versus)", mo.FEATS + ["h2h_diff"]),
                  ("  + form_diff (forma 20)", mo.FEATS + ["form_diff"]),
                  ("  + versus + forma", mo.FEATS + ["h2h_diff", "form_diff"])]:
    ll, auc, acc = evalua(tr, te, cols)
    print(f"  {nom:34} {ll:>9.4f} {auc:>7.4f} {acc:>7.3f}")

# ---------- 3) barrido de ventana de forma reciente ----------
print("\n" + "="*68); print("(3) % ganados últimos X: ¿alguna ventana aporta sobre Elo?"); print("="*68)
print(f"  {'ventana':>8} {'logloss base+form':>18} {'p-value de form_diff':>22}")
for n in [5, 10, 20, 40]:
    Dn = dataset(n)
    trn, ten = Dn[Dn.fecha < "2023-01-01"], Dn[Dn.fecha >= "2023-01-01"]
    ll, _, _ = evalua(trn, ten, mo.FEATS + ["form_diff"])
    Xn = pd.DataFrame(StandardScaler().fit_transform(trn[mo.FEATS + ["form_diff"]]), columns=mo.FEATS + ["form_diff"])
    pv = sm.Logit(trn.y.values, Xn).fit(disp=0).pvalues["form_diff"]
    print(f"  últimos {n:>2}  {ll:>17.4f} {pv:>22.4f}")

# ---------- 4) Elo vs full-stats SIN Elo ----------
print("\n" + "="*68); print("(4) Modelo del Streamlit (con Elo) vs FULL STATS SIN Elo"); print("="*68)
mo.FORM_N = 20
D = dataset(20); tr, te = D[D.fecha < "2023-01-01"], D[D.fecha >= "2023-01-01"]
sin_elo = [c for c in mo.CANDIDATAS if "elo" not in c]            # 12 vars: rank, edad, h2h, saque, resto, forma
solo_stats = [c for c in sin_elo if c not in ("log_rank_diff", "age_diff", "h2h_diff")]  # solo stats de juego
print(f"  {'modelo':40} {'logloss':>9} {'auc':>7} {'acc':>7}")
for nom, cols in [("Streamlit (Elo gen+sup+edad+rank)", mo.FEATS),
                  ("FULL STATS sin Elo (12 vars)", sin_elo),
                  ("Solo stats de juego sin Elo (9 vars)", solo_stats),
                  ("Solo Elo (gen+sup)", ["elo_overall_diff", "elo_surf_diff"])]:
    ll, auc, acc = evalua(tr, te, cols)
    print(f"  {nom:40} {ll:>9.4f} {auc:>7.4f} {acc:>7.3f}")
print(f"  (variables sin_elo: {sin_elo})")
