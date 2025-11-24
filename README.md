Indian Skincare Ingredient Classifier — Demo

A Machine Learning–powered tool to analyze skincare ingredient lists and predict safety categories.

This project uses Natural Language Processing (NLP) and Machine Learning to classify skincare ingredient lists into:

Safe for Sensitive Skin

Contains Mild Irritants

Comedogenic Risk

It is designed as an educational demo to help users understand ingredient profiles and potential sensitivities based on simple text analysis.

✨ Features

✅ Classifies ingredient lists using a trained ML pipeline
✅ Shows confidence scores for each category
✅ Flags common irritants & comedogenic ingredients
✅ Displays top contributing tokens (explainability)
✅ Beautiful Streamlit UI
✅ Deployed using Streamlit Cloud

🔍 How It Works

The core model is built using:

TF-IDF Vectorizer (text processing)

LinearSVC / Logistic Regression classifier

Label Encoding

Custom cleaning pipeline

Input ingredient lists are cleaned using standard INCI formatting rules and then passed through the ML pipeline for prediction.

Flagged ingredient categories include:

Irritants: fragrance compounds, sulfates, alcohols, etc.

Comedogenic Ingredients: coconut oil, lanolin, isopropyl myristate, etc.

🧠 Model Training

The model was trained on a small curated dataset labeled into three categories:

Comedogenic Risk

Contains Mild Irritants

Safe for Sensitive Skin

It uses TF-IDF features + linear classifier.
