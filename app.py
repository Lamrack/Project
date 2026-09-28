# ============================================================
# Cost-Aware Credit Card Fraud Detection — Streamlit tool
# Objective 7: interactive demonstration of cost-based
# threshold optimisation and the model crossover finding.
#
# Run locally:  streamlit run app.py
# Requires (same folder): histgb_smote.pkl, rf_none.pkl,
#                         scaler.pkl, feature_names.pkl
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---- Cost model (identical to the notebook) -----------------
C_ADMIN = 20.0   # fixed admin cost per missed fraud (£)

def expected_cost(y_true, y_pred, amounts, c_decline, c_admin=C_ADMIN):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); amounts = np.asarray(amounts)
    fn_mask = (y_true == 1) & (y_pred == 0)      # missed fraud
    fp_mask = (y_true == 0) & (y_pred == 1)      # false decline
    return np.sum(amounts[fn_mask] + c_admin) + c_decline * np.sum(fp_mask)

def best_threshold(y_true, scores, amounts, c_decline):
    """Threshold that minimises expected cost on the given data."""
    ts = np.linspace(0.001, 0.999, 999)
    costs = [expected_cost(y_true, (scores >= t).astype(int), amounts, c_decline) for t in ts]
    i = int(np.argmin(costs))
    return ts[i], costs[i]

# ---- Load artefacts (cached so it happens once) -------------
@st.cache_resource
def load_artefacts():
    histgb = joblib.load("histgb_smote.pkl")
    rf     = joblib.load("rf_none.pkl")
    scaler = joblib.load("scaler.pkl")
    feats  = joblib.load("feature_names.pkl")
    return histgb, rf, scaler, feats

def preprocess(df, scaler, feats):
    """Mirror the training pipeline: scale Time+Amount, align columns.
       Returns (X_scaled, amounts_in_pounds, y_or_None)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    y = df["Class"].values if "Class" in df.columns else None
    amounts = df["Amount"].values.astype(float)          # real £ BEFORE scaling
    X = df.reindex(columns=feats)                        # exact training columns/order
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
    return X, amounts, y

# ============================================================
st.set_page_config(page_title="Cost-Aware Fraud Detection", layout="wide")
st.title("Cost-Aware Credit Card Fraud Detection")
st.caption("Interactive demonstration of cost-based threshold optimisation. "
           "Reported thesis metrics come from the notebook's clean train/val/test "
           "pipeline; this tool is a demonstration layer.")

histgb, rf, scaler, feats = load_artefacts()

# ---- Sidebar: cost assumptions ------------------------------
st.sidebar.header("Cost assumptions")
c_decline = st.sidebar.slider("False-decline cost, C_decline (£)", 1, 100, 20, 1)
st.sidebar.write(f"Fixed admin cost per missed fraud: £{C_ADMIN:.0f}")
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload transactions CSV", type="csv")
st.sidebar.caption("CSV needs the 30 feature columns (Time, V1–V28, Amount). "
                   "Include a 'Class' column to see full cost analysis.")

if uploaded is None:
    st.info("Upload a CSV to begin. A labelled CSV (with a 'Class' column) "
            "unlocks the cost and crossover analysis below.")
    st.stop()

# ---- Process upload -----------------------------------------
raw = pd.read_csv(uploaded)
X, amounts, y = preprocess(raw, scaler, feats)

hg_scores = histgb.predict_proba(X)[:, 1]
rf_scores = rf.predict_proba(X)[:, 1]

if y is None:
    st.warning("No 'Class' column found — showing fraud scores only, "
               "no cost analysis possible.")
    out = raw.copy()
    out["HistGB_fraud_prob"] = hg_scores
    out["RF_fraud_prob"] = rf_scores
    st.dataframe(out.head(50))
    st.download_button("Download scored CSV", out.to_csv(index=False),
                       "scored.csv", "text/csv")
    st.stop()

# ---- Labelled path: full cost analysis at current C_decline --
hg_t, hg_cost = best_threshold(y, hg_scores, amounts, c_decline)
rf_t, rf_cost = best_threshold(y, rf_scores, amounts, c_decline)
winner = "HistGB + SMOTE" if hg_cost <= rf_cost else "RandomForest (no balancing)"

st.subheader(f"At C_decline = £{c_decline}: cheapest model is **{winner}**")
col1, col2 = st.columns(2)
for col, name, scores, thr, cost in [
    (col1, "HistGB + SMOTE", hg_scores, hg_t, hg_cost),
    (col2, "RandomForest (no balancing)", rf_scores, rf_t, rf_cost)]:
    with col:
        st.markdown(f"### {name}")
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        st.metric("Total expected cost", f"£{cost:,.2f}")
        st.write(f"Cost-optimal threshold: **{thr:.3f}**")
        st.write(f"Fraud caught (recall): **{tp}/{tp+fn}** ({tp/(tp+fn):.1%})")
        st.write(f"False declines: **{fp}**  |  Missed frauds: **{fn}**")

# ---- The crossover plot: cost vs C_decline for both models ---
st.markdown("---")
st.subheader("Sensitivity: how cost depends on the false-decline assumption")
c_range = np.arange(1, 101, 1)
hg_curve, rf_curve = [], []
for c in c_range:
    t1, _ = best_threshold(y, hg_scores, amounts, c)
    t2, _ = best_threshold(y, rf_scores, amounts, c)
    hg_curve.append(expected_cost(y, (hg_scores >= t1).astype(int), amounts, c))
    rf_curve.append(expected_cost(y, (rf_scores >= t2).astype(int), amounts, c))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(c_range, hg_curve, label="HistGB + SMOTE", linewidth=2)
ax.plot(c_range, rf_curve, label="RandomForest (no balancing)", linewidth=2)
ax.axvline(c_decline, color="grey", linestyle="--", label=f"Current (£{c_decline})")
ax.set_xlabel("False-decline cost, C_decline (£)")
ax.set_ylabel("Total expected cost (£)")
ax.set_title("Model cost across the false-decline range")
ax.legend(); ax.grid(alpha=0.3)
st.pyplot(fig)
st.caption("Where the lines cross, the cheaper model changes — the crossover finding.")
