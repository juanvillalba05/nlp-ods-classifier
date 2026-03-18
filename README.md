# NLP Classifier for UN Sustainable Development Goals (SDGs)

Automatic text classification system that maps any document to one of the
17 UN Sustainable Development Goals (SDGs) using natural language processing
and supervised machine learning.

## Overview

Developed as a contribution to the analysis of public policy documents,
citizen participation texts, and reports in the context of the UN 2030 Agenda.
The model processes Spanish-language texts and predicts which SDG they relate to,
enabling faster and more scalable analysis than manual expert review.

## Pipeline
```
Raw text → TF-IDF (BOW) → TruncatedSVD (LSA) → Logistic Regression → SDG prediction
```

## Key steps

- **Dataset:** OSDG Community Dataset v2023 — 40,000+ labeled texts in Spanish,
  covering all 17 SDGs (UN documents, article abstracts, public reports)
- **Preprocessing:** Spanish stopword removal (NLTK), TF-IDF vectorization
  with `min_df` and `max_df` filtering
- **Dimensionality reduction:** TruncatedSVD (Latent Semantic Analysis)
  with 100 components — chosen for its efficiency with sparse TF-IDF matrices
- **Classifier:** Logistic Regression with hyperparameter tuning via GridSearchCV
- **Evaluation:** Accuracy, classification report, and confusion matrix on a
  held-out test set

## Results

The model correctly classifies texts across all 17 SDG categories,
handling class imbalance present in the original dataset.

## Tech stack

- Python
- scikit-learn (TfidfVectorizer, TruncatedSVD, LogisticRegression, GridSearchCV, Pipeline)
- NLTK
- NumPy, pandas
- Matplotlib, Seaborn
- Streamlit (interactive demo app)
- Jupyter Notebook

## How to run
```bash
git clone https://github.com/tu-usuario/nlp-ods-classifier
cd nlp-ods-classifier
pip install -r requirements.txt
jupyter notebook nlp-ods-classifier.ipynb
```

## Streamlit app

The model is also deployed as an interactive web app where the user can
enter any free text and get an instant SDG prediction:
```bash
streamlit run app.py
```

## Context

Project developed as part of the Unsupervised Machine Learning course —
Master's in Artificial Intelligence.

**Dataset:** [OSDG Community Dataset](https://osdg.ai/news/New-release-of-OSDG-Community-dataset)
```
