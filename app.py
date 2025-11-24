import streamlit as st
import joblib
import pandas as pd
import re
import os

MODEL_PATH = "ingredient_classifier_v1.joblib"
CM_PATH = "confusion_matrix.png"

@st.cache_resource
def load_model(path):
    data = joblib.load(path)
    return data["pipeline"], data["label_encoder"]

try:
    pipe, le = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

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

st.set_page_config(page_title="Skincare Ingredient Classifier")
st.title("Indian Skincare Ingredient Classifier — Demo")

ing = st.text_area("Paste ingredient list:", height=200)

if st.button("Analyze"):
    cleaned = clean_ingredients(ing)

    if not cleaned:
        st.warning("Paste an ingredient list first.")
        st.stop()

    pred_idx = pipe.predict([cleaned])[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    probs = pipe.predict_proba([cleaned])[0]

    st.subheader(f"Prediction: **{pred_label}**")
    st.write({le.classes_[i]: float(round(probs[i],3)) for i in range(len(probs))})

    flagged = []
    for item in FLAG_IRRITANTS:
        if item in cleaned:
            flagged.append(("Irritant", item))

    for item in FLAG_COMEDO:
        if item in cleaned:
            flagged.append(("Comedogenic", item))

    if flagged:
        st.warning("Flagged ingredients:")
        for cat, name in flagged:
            st.write(f"- **{name}** — {cat}")
    else:
        st.success("No flagged irritants/comedo ingredients found.")

st.caption("Disclaimer: Educational demo only — not medical advice.")
