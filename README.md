🌿 Indian Skincare Ingredient Classifier — Demo

A Machine Learning–powered tool to analyze skincare ingredient lists and predict safety categories.

This project uses Natural Language Processing (NLP) + Machine Learning to classify ingredient lists into:

Safe for Sensitive Skin

Contains Mild Irritants

Comedogenic Risk

Designed as an educational demo to help users understand ingredient profiles and potential sensitivities based on simple text analysis.

🌐 Live Demo

Try the deployed app here:

👉 https://skincare-ingredient-classifier-yrpfsm8mrcnnpscg4vdx9s.streamlit.app

✨ Features

✅ Classifies ingredient lists using a trained ML pipeline

✅ Displays confidence scores for each category

✅ Flags common irritants & comedogenic ingredients

✅ Shows top contributing tokens (explainability)

✅ Clean, simple Streamlit UI

✅ Deployed using Streamlit Cloud

🔍 How It Works

The core model is built using:

TF-IDF Vectorizer (text processing)

LinearSVC / Logistic Regression classifier

Label Encoding

Custom cleaning pipeline

Ingredient lists are cleaned using standard INCI formatting rules and passed through the ML pipeline for prediction.

Flagged ingredient categories:

Irritants:

fragrance compounds

sulfates

alcohols

limonene, linalool

etc.

Comedogenic ingredients:

coconut oil

lanolin

isopropyl myristate

mineral oil

shea butter

etc.

🧠 Model Training

The model was trained on a curated dataset labeled into three categories:

Comedogenic Risk

Contains Mild Irritants

Safe for Sensitive Skin

It uses TF-IDF features + a Linear classifier.
