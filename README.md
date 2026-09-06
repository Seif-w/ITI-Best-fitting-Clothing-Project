# ITI-Best-fitting-Clothing-Project

An AI-powered system that predicts whether a clothing item will **fit** a given user, by combining computer vision, NLP, and tabular ML on user reviews, body measurements, and product images.

This project integrates three types of signals:
- **Visual data** — what the garment looks like (via transfer learning on a pretrained CNN)
- **Textual data** — what users say about fit in their reviews (via NLP)
- **Numerical/tabular data** — user measurements (height, weight, age) and product metadata (size, category, brand)

...into a single fused feature set used to train a model that predicts a **fit score (0–1)**.

> This project is for research, experimentation, and model development — it is not intended for production deployment.

---

## Table of Contents
- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
- [Pipeline](#pipeline)
- [Datasets](#datasets)
- [Known Limitation (Read This First)](#known-limitation-read-this-first)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Evaluation Metrics](#evaluation-metrics)
- [Team & Roles](#team--roles)
- [Future Enhancements](#future-enhancements)

---

## Motivation

Online apparel returns run as high as 30–40%, mostly due to sizing and fit issues. Unstructured data (images, reviews) makes this hard to model with traditional ML alone. This project takes a multi-model approach — combining Computer Vision, NLP, and Machine Learning — to produce a fit score per user-item interaction, aiming to reduce returns and improve recommendation quality.

---

## Architecture Overview

```
                        ┌───────────────────────────┐
                        │        Data Layer         │
                        │  (RTR / ModCloth dataset,  │
                        │   Fashion Product Images)  │
                        └─────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
    ┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
    │  Computer Vision   │   │        NLP         │   │   Numeric/Meta    │
    │      Pipeline      │   │      Pipeline       │   │    Preprocessing   │
    │ (Pretrained CNN /  │   │ (Text cleaning +    │   │ (StandardScaler,   │
    │  ResNet, frozen)   │   │  TF-IDF / BERT)      │   │  OneHotEncoder)    │
    └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                          ┌───────────▼───────────┐
                          │     Feature Fusion      │
                          │  (Concatenate all into  │
                          │     one feature row)     │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │      ML/DL Model         │
                          │ (XGBoost / Keras Dense)  │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   Fit Score (0 – 1)      │
                          └───────────────────────┘
```

---

## Pipeline

### 1. Category Filtering
Filter the Fashion Product Images dataset down to categories matching RTR/ModCloth (dress, gown, romper, sheath, etc.). Because there are no real photos of the actual reviewed items, each category is represented by a *group* of images rather than a single one.

### 2. Visual Feature Extraction (CNN)
Use a pretrained CNN (ResNet50/ResNet101) via transfer learning. The backbone is **frozen** (pure feature extraction, not fine-tuning) — the output before the classification layer is used as a feature vector (2048-dim, via `GlobalAveragePooling2D`).

### 3. Category-Level Vector Aggregation
Since there's no one-to-one image per row, all feature vectors belonging to the same category are **averaged** into a single vector representing the general visual appearance of that category (e.g., "what dresses generally look like").

### 4. Join Category Vector to Main Dataset
The RTR/ModCloth table is joined with the category-vector table on the `category` column, so every user review inherits its category's visual vector.

### 5. Text Feature Extraction (NLP)
Reviews are cleaned (regex cleaning → tokenization → case folding → stop-word removal → lemmatization/stemming) and vectorized via **TF-IDF** (or BERT, if the team can defend the added complexity — see [Known Limitation](#known-limitation-read-this-first)).

### 6. Numeric & Categorical Preprocessing
Numeric columns (height, weight, age) are standardized with `StandardScaler`; categorical columns (size, brand) are one-hot encoded.

### 7. Fusion
Visual vector + text vector + numeric/categorical features are concatenated into a single long feature row per user-item interaction.

### 8. Model Training
The fused feature matrix is fed into a final model — **XGBoost** (tabular-friendly, handles mixed feature types well) or a **Keras Dense network** — trained to predict the fit score.

### 9. Evaluation
Since fit score is continuous (not a fixed class), evaluation uses **MAE, RMSE, R²** rather than accuracy. If fit is instead bucketed into classes (small/fit/large), classification metrics (confusion matrix, precision/recall/F1, ROC-AUC) apply instead.

### 10. Limitation Documentation
The visual feature operates at the **category level, not the item level**, due to the absence of real per-item images. This is documented explicitly as a deliberate, acknowledged design trade-off — not an oversight.

---

## Datasets

| Component | Dataset | Use Case |
|---|---|---|
| Fit labels & reviews | Rent the Runway (RTR) / ModCloth |[ Core fit prediction dataset](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation?resource=download) |
| Computer Vision | Fashion Product Images (Kaggle) |[ Category-level visual features ](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)|


---

## Known Limitation (Read This First)

Two decisions in this pipeline are **workarounds, not ideal solutions** — flagged here deliberately so the team (and reviewers) go in with eyes open:

1. **Category-level visual features.** Every item in a given category (e.g., every "dress") receives the *same* averaged visual vector. The model can distinguish between categories (dress vs. romper) but **cannot** distinguish between two different dresses. This caps how much the visual signal can help.
2. **NLP method choice.** TF-IDF is the safer, fully-explainable choice given the team's study material. BERT is more powerful but requires the team to independently understand and defend it, since it isn't covered in the group's core reference material.

---

## Tech Stack

- **Languages:** Python
- **Computer Vision:** TensorFlow/Keras, ResNet50/ResNet101 (pretrained, ImageNet weights)
- **NLP:** spaCy / NLTK (cleaning), scikit-learn (TF-IDF) or Transformers (BERT, optional)
- **Modeling:** XGBoost, scikit-learn, TensorFlow/Keras
- **Preprocessing:** scikit-learn (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`)
- **Evaluation:** scikit-learn (`mean_absolute_error`, `mean_squared_error`, `r2_score`, `classification_report`)
- **Data handling:** pandas, NumPy

---

## Repository Structure

```
ITI-Best-fitting-Clothing-Project/
├── data/
│   ├── raw/                  # Original RTR/ModCloth + Fashion Product Images
│   └── processed/            # Cleaned, filtered, merged datasets
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_cv_feature_extraction.ipynb
│   ├── 03_nlp_feature_extraction.ipynb
│   ├── 04_fusion_and_preprocessing.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── data/                 # Filtering, category mapping, merging
│   ├── cv/                   # CNN feature extraction, category averaging
│   ├── nlp/                  # Text cleaning, TF-IDF/BERT vectorization
│   ├── fusion/                # Scaling, encoding, concatenation
│   ├── models/                # Training scripts (XGBoost / Keras)
│   └── evaluation/            # Metrics and error analysis
├── reports/
│   └── final_report.md        # Methodology, results, limitations
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
git clone https://github.com/<org>/ITI-Best-fitting-Clothing-Project.git
cd ITI-Best-fitting-Clothing-Project
pip install -r requirements.txt
```

Basic `requirements.txt` should include:
```
tensorflow
scikit-learn
xgboost
pandas
numpy
spacy
nltk
matplotlib
seaborn
```

---

## Evaluation Metrics

| Metric | Used For | Why |
|---|---|---|
| MAE / RMSE / R² | Fit score regression | Fit score is a continuous 0–1 value; accuracy is not a valid metric here |
| Precision / Recall / F1 | Fit classification (if bucketed) | Needed if fit is treated as small/fit/large classes instead of a continuous score |
| Confusion Matrix / ROC-AUC | Fit classification (if bucketed) | Visualizes class-level performance and threshold sensitivity |

---

## Team & Roles

| # | Role | Responsibility |
|---|---|---|
| 1 | Data Lead | RTR/ModCloth cleaning, category taxonomy mapping between datasets |
| 2 | Computer Vision Engineer | CNN transfer learning, category-vector averaging, visual limitation write-up |
| 3 | NLP Engineer | Text cleaning pipeline, TF-IDF/BERT decision and implementation |
| 4 | Feature Fusion Engineer | Scaling, encoding, concatenation into final feature matrix |
| 5 | Modeling Lead | Model training (XGBoost/Keras), hyperparameter tuning |
| 6 | Evaluation & Validation | Metrics, cross-validation, error analysis |
| 7 | Integration & Documentation Lead | End-to-end pipeline assembly, final report, presentation |

---

## Future Enhancements

- 3D body scanning for more accurate measurements
- Real-time virtual try-on using AR
- Multilingual NLP for global personalization
- Social media trend integration
- Continual learning from user feedback
