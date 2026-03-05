# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions.  
Trained on a dataset with a 559:1 class imbalance using XGBoost with threshold optimization.

## Results

| Metric | Validation | Test |
|--------|------------|------|
| F1-Score | 0.8757 | 0.8646 |
| PR-AUC | 0.8527 | 0.8620 |
| ROC-AUC | 0.9869 | 0.9736 |

## Project Structure
```
credit_fraud_project/
├── data/
│   └── raw/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── notebooks/
│   ├── EDA.ipynb
│   ├── feature_engineering.ipynb
│   └── sampling_model_selection.ipynb
├── outputs/
│   ├── models/
│   ├── figures/
│   └── reports/
├── src/
│   ├── models/
│   │   ├── logistic_model.py
│   │   ├── rf_model.py
│   │   ├── xgb_model.py
│   │   ├── nn_model.py
│   │   └── voting_model.py
│   ├── config.py
│   ├── feature_engineering.py
│   ├── credit_fraud_train.py
│   ├── credit_fraud_evaluate.py
│   ├── credit_fraud_utils_data.py
│   ├── credit_fraud_utils_eval.py
│   └── credit_fraud_utils_sampling.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train best model
python src/credit_fraud_train.py

# Evaluate on Test set
python src/credit_fraud_evaluate.py
```

## Approach

**Feature Engineering**
- `log_amount` — reduces skewness in Amount from 19.99 to 0.16
- `time_sin`, `time_cos` — cyclical encoding of Time

**Model Selection**  
Seven sampling strategies × five models were evaluated on the Validation set.  
XGBoost with no resampling achieved the best F1-Score — `scale_pos_weight=559.28` was sufficient to handle the class imbalance.

**Threshold Optimization**  
Optimal threshold derived from the Precision-Recall Curve by maximizing F1-Score.  
A fixed threshold of 0.5 is inappropriate for imbalanced datasets.

| Decision | Choice | Reason |
|----------|--------|--------|
| Primary metric | F1-Score | Balances precision and recall |
| Imbalance handling | scale_pos_weight=559.28 | Sufficient without resampling |
| Threshold | PR Curve (max F1) | Outperforms fixed threshold |
| Calibration | Isotonic regression | Corrects XGBoost probability overconfidence |
| Feature selection | SelectFromModel (median) | Removes low-importance features |
| Cross-validation | StratifiedKFold (5 folds) | Preserves fraud ratio across folds |
