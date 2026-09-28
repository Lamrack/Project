# Cost-Aware Credit Card Fraud Detection

An interactive tool demonstrating cost-sensitive threshold optimisation for
credit card fraud detection under extreme class imbalance. Developed as part of
a BSc final-year project.

## What it does

Fraud detection is dominated by extreme class imbalance (fraud is ~0.17% of
transactions in the ULB dataset). Standard accuracy is misleading in this
setting, and the two error types carry very different costs: a missed fraud
loses the transaction amount plus fixed admin/chargeback costs, while a false
decline imposes a fixed customer-friction cost.

This tool lets a user vary the assumed cost of a false decline and see, live:

- the cost-optimal decision threshold at that cost,
- the resulting confusion matrix, recall, and total expected cost,
- which of two trained models is cheaper at that cost — demonstrating that the
  best model choice depends on the cost assumption (the "crossover" finding).

## Models

Two models trained on the ULB Credit Card Fraud dataset are included:

- `histgb_smote.pkl` — Histogram Gradient Boosting with SMOTE (most robust
  across the cost range).
- `rf_none.pkl` — Random Forest without resampling (cheaper in a narrow
  low-cost band).

Thresholds are chosen by minimising expected cost on a validation set;
reported project metrics come from a clean train/validation/test pipeline.
This tool is a demonstration layer over the saved models.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown (usually http://localhost:8501).

## Input format

Upload a CSV with the 30 feature columns (`Time`, `V1`–`V28`, `Amount`).
Include a `Class` column (0 = legitimate, 1 = fraud) to unlock the full cost
and crossover analysis against ground-truth labels.

## Files

| File | Purpose |
|------|---------|
| `app.py` | The Streamlit application |
| `histgb_smote.pkl` | Trained HistGB + SMOTE model |
| `rf_none.pkl` | Trained Random Forest model |
| `scaler.pkl` | StandardScaler fitted on training Time/Amount |
| `feature_names.pkl` | Expected feature order |
| `requirements.txt` | Pinned dependencies |

## Dataset

ULB Credit Card Fraud Detection dataset (Dal Pozzolo et al.), 284,807
anonymised transactions with PCA-transformed features. Publicly available on
Kaggle. Not redistributed in this repository.

## Note

This is academic coursework. The tool is a research demonstration, not a
production fraud-detection system.
