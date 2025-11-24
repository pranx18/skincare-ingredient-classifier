# app.py  — Redesigned compact UI for Skincare Ingredient Classifier
import streamlit as st
import joblib
import pandas as pd
import re
import os
import numpy as np

# ---------- Config ----------
MODEL_PATH = "ingredient_classifier_v1.joblib"
CM_PATH = "confusion_matrix.png"     # optional
DATA_URL = "/skincare_dataset1.csv"  # your uploaded dataset local path

# ---------- Page / CSS ----------
st.set_page_config(page_title="Skincare Ingredient Classifier", layout="wide")

# compact card styles
st.markdown(
    """
    <style>
      /* page */
      .stApp {
        background-color: #ffffff;
      }
      /* card look for major panels */
      .card {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        padding: 18px;
        margin-bottom: 12px;
      }
      /* headings */
      .big-title { font-size:28px; font-weight:700; color:#0f172a; margin-bottom:6px; }
      .muted { color:#64748b; font-size:14px; }
      /* smaller text */
      .compact { margin:0; padding:0; font-size:14px; color:#0f172a; }
      /* button accent */
      .stButton>button {
        background: linear-gradient(90deg,#0ea5a3,#06b6d4);
        color: white;
        border: none;
      }
      /* small caption style */
      .caption { font-size:12px; color:#94a3b8; }
      /* remove large gaps from Streamlit containers */
      .element-container { padding-top: 0.25rem; padding-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model ----------
@st.cache_resource
def load_model(path):
    data = joblib.load(path)
    return data["pipeline"], data["label_encoder"]

try:
    pipe, le = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model from '{MODEL_PATH}': {e}")
    st.stop()

# ---------- Helpers ----------
def clean_ingredients(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[/\\n]', ',', text)
    text = re.sub(r'[^a-z0-9, ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r',\s*,', ',', text)
    return text.strip(' ,')

FLAG_IRRITANTS = [
    "fragrance","parfum","limonene","linalool",
    "sodium laureth sulfate","sodium lauryl sulfate",
    "alcohol denat","cocamide dea","cocamide mea"
]
FLAG_COMEDO = [
    "coconut oil","isopropyl myristate","lanolin",
    "mineral oil","wheat germ oil","almond oil","shea butter"
]

# ---------- Header ----------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center">', unsafe_allow_html=True)
    st.markdown('<div><h1 class="big-title">Skincare Ingredient Classifier</h1>'
                '<div class="muted">Paste an INCI ingredient list and get a classification (demo)</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:right"><div class="caption">Dataset (local): <code>{DATA_URL}</code></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Main layout: input (left) / results (right) ----------
left, right = st.columns([2.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<strong class="compact">Paste ingredient list (INCI)</strong>', unsafe_allow_html=True)
    user_input = st.text_area("", height=220, placeholder="Aqua, Glycerin, Niacinamide, ...", key="input_area")
    # quick action row
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        analyze = st.button("Analyze")
    with col2:
        # example dropdown for quick loads
        example = st.selectbox("Quick examples", ["— select example —", "Gentle serum", "Rich cream"], key="example_box")
        if example == "Gentle serum":
            user_input = "Aqua, Glycerin, Niacinamide, Panthenol, Sodium Hyaluronate"
            st.session_state["input_area"] = user_input
        elif example == "Rich cream":
            user_input = "Aqua, Mineral Oil, Shea Butter, Lanolin"
            st.session_state["input_area"] = user_input
    with col3:
        # model download
        try:
            with open(MODEL_PATH, "rb") as f:
                model_bytes = f.read()
            st.download_button("Download model", data=model_bytes, file_name=MODEL_PATH)
        except Exception:
            st.info("Model file not in repo for download.")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<strong class="compact">Model Info</strong>', unsafe_allow_html=True)
    st.write("Pipeline: TF-IDF → LogisticRegression (demo)")
    if os.path.exists(CM_PATH):
        st.image(CM_PATH, caption="Confusion matrix", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Analysis & Results ----------
if analyze and user_input.strip():
    cleaned = clean_ingredients(user_input)
    pred_idx = pipe.predict([cleaned])[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    probs = pipe.predict_proba([cleaned])[0]

    # Results card
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Top row: prediction + badge
    st.markdown(f'<div style="display:flex; align-items:center; gap:16px;"><h2 style="margin:0">{pred_label}</h2>', unsafe_allow_html=True)
    # confidence little badge
    best_prob = float(round(np.max(probs), 3))
    st.markdown(f'<div style="background:linear-gradient(90deg,#06b6d4,#0ea5a3); color:white; padding:6px 12px; border-radius:999px; font-weight:600">{best_prob*100:.0f}%</div></div>', unsafe_allow_html=True)

    # bar chart for confidences (compact)
    confidences = pd.Series(probs, index=le.classes_)
    confidences = confidences.sort_values(ascending=True)
    st.bar_chart(confidences)

    # token contributions (approx) as small table
    try:
        vec = pipe.named_steps['tfidf']
        clf = pipe.named_steps['clf']
        feature_names = vec.get_feature_names_out()
        x = vec.transform([cleaned]).toarray()[0]
        contributions = {}
        for i, class_label in enumerate(le.classes_):
            contrib = x * clf.coef_[i]
            top_idx = contrib.argsort()[-6:][::-1]
            top = [(feature_names[j], float(round(contrib[j],6))) for j in top_idx if x[j] > 0]
            contributions[class_label] = top
        st.markdown("**Top tokens (approx)**")
        for cls, toks in contributions.items():
            st.write(f"- **{cls}**: {', '.join([t[0] for t in toks]) if toks else '—'}")
    except Exception:
        pass

    # Flags & explanation column
    st.markdown('<div style="display:flex; gap:12px; margin-top:12px;">', unsafe_allow_html=True)
    # left side: flagged
    st.markdown('<div style="flex:1">', unsafe_allow_html=True)
    st.markdown("**Flags**")
    flagged = []
    lc = cleaned.lower()
    for item in FLAG_IRRITANTS:
        if item in lc:
            flagged.append(("Irritant", item))
    for item in FLAG_COMEDO:
        if item in lc:
            flagged.append(("Comedogenic", item))

    if flagged:
        for cat, ing_name in flagged:
            st.warning(f"{ing_name} — {cat}")
    else:
        st.success("No common flagged irritants/comedogenic ingredients found (simple check).")
    st.markdown('</div>', unsafe_allow_html=True)

    # right side: confidence table
    st.markdown('<div style="flex:1">', unsafe_allow_html=True)
    st.markdown("**Confidence breakdown**")
    conf_df = pd.DataFrame({"class": le.classes_, "probability": list(probs)})
    st.table(conf_df)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close flags row
    st.markdown('</div>', unsafe_allow_html=True)  # close card

# Footer
st.markdown("---")
st.caption("This demo is educational and not medical advice. Patch-test and consult a dermatologist for personalised concerns.")
