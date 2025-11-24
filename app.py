# app.py - Streamlit demo (ready for Streamlit Cloud)
import streamlit as st
import joblib
import pandas as pd
import re
import os

# ---- Files (assume placed in repo root) ----
MODEL_PATH = "ingredient_classifier_v1.joblib"
CM_PATH = "confusion_matrix.png"   # optional - include if you want to show it

# ---- Load model once ----
@st.cache_resource
def load_model(path):
    data = joblib.load(path)
    return data["pipeline"], data["label_encoder"]

try:
    pipe, le = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model from '{MODEL_PATH}': {e}")
    st.stop()

# ---- cleaning helper (same as used in training) ----
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

# ---- simple flagged ingredient lists (expand as needed) ----
FLAG_IRRITANTS = [
    "fragrance","parfum","limonene","linalool",
    "sodium laureth sulfate","sodium lauryl sulfate",
    "alcohol denat","cocamide dea","cocamide mea"
]
FLAG_COMEDO = [
    "coconut oil","isopropyl myristate","lanolin",
    "mineral oil","wheat germ oil","almond oil","shea butter"
]

# ---- Streamlit UI ----
st.set_page_config(page_title="Skincare Ingredient Classifier", layout="centered")
st.title("Indian Skincare Ingredient Classifier — Demo")
st.markdown(
    "Paste an *INCI-style* ingredient list and get a prediction (educational demo). "
    "Categories: **Safe for Sensitive Skin**, **Contains Mild Irritants**, **Comedogenic Risk**."
)

ing = st.text_area("Paste ingredient list (INCI)", height=200, placeholder="Aqua, Glycerin, Niacinamide, ...")

if st.button("Analyze"):
    cleaned = clean_ingredients(ing)
    if not cleaned:
        st.warning("Please paste an ingredient list before analyzing.")
    else:
        pred_idx = pipe.predict([cleaned])[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        probs = pipe.predict_proba([cleaned])[0]
        st.subheader(f"Prediction: {pred_label}")
        st.write("Confidence:", {le.classes_[i]: float(round(probs[i],3)) for i in range(len(probs))})

        # flagged ingredient simple check
        flagged = []
        lc = cleaned.lower()
        for item in FLAG_IRRITANTS:
            if item in lc:
                flagged.append(("Irritant", item))
        for item in FLAG_COMEDO:
            if item in lc:
                flagged.append(("Comedogenic", item))

        if flagged:
            st.warning(⚠️ Flagged ingredients found:")
            for cat, ing_name in flagged:
                st.write(f"- **{ing_name}** — {cat}")
        else:
            st.success("No common flagged irritants/comedogenic ingredients found (simple substring check).")

        # optional: show top tokens (approx) using coef * tfidf
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
            st.markdown("**Approx top contributing tokens (token, contribution)**")
            for cls, toks in contributions.items():
                st.write(f"- {cls}: {', '.join([t[0] for t in toks]) if toks else '—'}")
        except Exception:
            pass

# optional: show confusion matrix if included in repo
if os.path.exists(CM_PATH):
    st.markdown("---")
    st.subheader("Model Confusion Matrix (training)")
    st.image(CM_PATH, use_column_width=True)

st.markdown("---")
st.caption("**Disclaimer:** Educational demo — not medical advice. Always patch-test and consult a dermatologist for personal concerns.")
