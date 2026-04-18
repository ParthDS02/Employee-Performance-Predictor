"""
Predict Employee Performance — Streamlit App
Author : Parth B Mistry

Lightweight UI-only file for deployment.
Model training happens offline using train_model.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Employee Performance Predictor", page_icon="👔", layout="centered")

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler_new.pkl"
LABELS      = {1: "🔴 Low", 2: "🟡 Average", 3: "🟢 Good", 4: "🟢 High", 5: "🏆 Excellent"}

# ── Load Pre-trained Model ─────────────────────────────────────────────────────
@st.cache_resource
def load_ml_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("❌ **Pre-trained model not found!**\nPlease run `python train_model.py` to generate `model.pkl` and `scaler_new.pkl`.")
        st.stop()
    
    with open(MODEL_PATH, "rb") as f:  model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    return model, scaler

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👔 Employee Performance")
    st.markdown("Predict performance using realistic HR patterns powered by Random Forest.")
    st.divider()
    st.markdown("**Algorithm:** Random Forest (balanced)")
    st.markdown("**Features:** 6 key HR metrics")
    st.markdown("**Tech Stack:** scikit-learn · Streamlit · Python")
    st.divider()
    st.markdown("Built by **Parth B Mistry**")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("👔 Employee Performance Predictor")
st.caption("ML-powered prediction using Random Forest • Deployed Version")

with st.spinner("Loading pre-trained model…"):
    model, scaler = load_ml_assets()

st.success("✅ Model loaded successfully from cache!")
st.divider()

# ── How it works ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ How does the model score performance?"):
    st.markdown("""
    | Factor | Weight | Insight |
    |--------|--------|---------|
    | Employee Satisfaction | **35%** | Happy employees consistently outperform |
    | Promotions | **25%** | Rewarded employees stay motivated |
    | Training Hours | **20%** | Skills directly improve output quality |
    | Projects Handled | **15%** | More projects = proven capability |
    | Years at Company | **5%** | Experience adds a small positive effect |
    """)

# ── Input panel ────────────────────────────────────────────────────────────────
st.subheader("🔢 Enter Employee Details")
col1, col2 = st.columns(2)
with col1:
    years      = st.slider("Years at Company",             0,   40,   5)
    salary     = st.number_input("Monthly Salary (₹ / $)", 2000, 100000, 6000, step=500)
    promotions = st.slider("Number of Promotions",         0,    6,   2)
with col2:
    satisf     = st.slider("Satisfaction Score (1–5)",     1.0, 5.0, 3.5, step=0.1)
    training   = st.slider("Training Hours",               0,  100, 30)
    projects   = st.slider("Projects Handled",             0,   50, 15)

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🚀 Predict Performance", use_container_width=True, type="primary"):
    raw    = np.array([[years, salary, promotions, satisf, training, projects]])
    scaled = scaler.transform(raw)
    pred   = int(model.predict(scaled)[0])
    proba  = model.predict_proba(scaled)[0]
    classes = model.classes_.tolist()

    label = LABELS.get(pred, str(pred))
    conf  = proba[classes.index(pred)] * 100 if pred in classes else 0.0

    st.markdown(f"### Predicted Performance: **{label}**")
    col_a, col_b = st.columns(2)
    col_a.metric("Predicted Class", label)
    col_b.metric("Model Confidence", f"{conf:.1f}%")

    st.subheader("📊 Probability Breakdown (All Classes)")
    chart_df = pd.DataFrame({
        "Level":            [LABELS[c] for c in classes],
        "Probability (%)":  (proba * 100).round(1)
    }).set_index("Level")
    st.bar_chart(chart_df)

    st.subheader("💡 Why this prediction?")
    reasons = []

    if satisf >= 4.0:   reasons.append("✅ **High Satisfaction:** Highly satisfied employees are the strongest performers.")
    elif satisf <= 2.5: reasons.append("⚠️ **Low Satisfaction:** Low morale acts as a major drag on performance.")
    
    if promotions >= 3: reasons.append("✅ **Frequent Promotions:** Consistent internal growth signals top-tier output.")
    elif promotions == 0 and years > 3: reasons.append("⚠️ **Stagnation:** 0 promotions over several years negatively impacts the score.")
    
    if training >= 60:  reasons.append("✅ **Extensive Training:** High investment in upskilling boosts prediction significantly.")
    elif training <= 15:reasons.append("⚠️ **Minimal Training:** Lack of recent upskilling hurts the overall score.")
    
    if projects >= 30:  reasons.append("✅ **High Project Output:** Handling massive project load strongly indicates 'Excellent' capacity.")

    if not reasons:
        reasons.append("ℹ️ **Balanced Profile:** The employee's metrics are average across the board, leaning neither extremely positive nor extremely negative.")

    if pred in [4, 5]: st.success("\n\n".join(reasons))
    elif pred == 3:    st.info("\n\n".join(reasons))
    else:              st.warning("\n\n".join(reasons))

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Employee Performance Predictor · Random Forest · Built by **Parth B Mistry**")