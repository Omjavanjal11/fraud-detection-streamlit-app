#  Fraud detection streamlit app

## Project Overview

An end-to-end fraud detection pipeline built on the IEEE-CIS Fraud Detection dataset.
The system combines state-of-the-art machine learning, severe class imbalance handling,
SHAP Explainable AI, and a live interactive Streamlit dashboard.

**Live Dashboard:** https://fraud-detection-app-app-ogqy7kwqsydmantwyappu3h.streamlit.app/

---

##  Project Structure

```
fraud-detection-streamlit-app/
│
├── app.py
├── model.pkl
├── analysis.ipynb
├── requirements.txt
├── README.md
├── model_comparison.png
├── shap_summary.png
```

##  Models Used

| Model | Type | Purpose |
|-------|------|---------|
| **LightGBM** | Gradient Boosting | Primary classifier (best performance) |
| **XGBoost** | Gradient Boosting | Comparison model |
| **Isolation Forest** | Unsupervised Anomaly Detection | Baseline comparison |

**Best Model:** LightGBM tuned with Optuna (30 trials, optimizing PR-AUC)

---

## Features

Real-time fraud prediction
Explainable AI using SHAP
Interactive Streamlit dashboard
Fraud risk segmentation
Threshold optimization
Multiple ML model comparison
Visual analytics & charts
Class imbalance handling using SMOTE

##  Key Results

- **PR-AUC:** Best measure for imbalanced fraud detection
- **Optimal Threshold:** Found via F1-Score maximization
- **SMOTE:** Applied only on training set to prevent data leakage
- **Risk Tiers:** 🔴 Critical (≥0.75) | 🟡 Suspicious (0.40–0.74) | 🟢 Clear (<0.40)

---


##  Tech Stack

`Python 3.10` | `LightGBM` | `XGBoost` | `scikit-learn` | `SHAP` | `Optuna`
`imbalanced-learn` | `Pandas` | `NumPy` | `Plotly` | `Streamlit` | `Matplotlib` | `Seaborn`

---


## Author
Om Javanjal
