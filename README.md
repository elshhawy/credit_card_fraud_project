# Credit-Card-Fraud-Detection

## Project Structure
```bash
CREDIT_FRAUD_PROJECT/
├── data/raw/              
├── notebooks/             ← EDA + experiments
├── outputs/
│   ├── models/            ← best_model.pkl
│   ├── figures/           ← plots
│   └── reports/           ← results CSV
├── src/
│   ├── models/
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
# 1. Train best model
python src/credit_fraud_train.py

# 2. Evaluate on Test set
python src/credit_fraud_evaluate.py
```

## Results
| Metric   | Validation | Test   |
|----------|------------|--------|
| F1-Score | 0.8757     | 0.8646 |
| PR-AUC   | 0.8527     | 0.8620 |
| ROC-AUC  | 0.9869     | 0.9736 |

## Key Decisions
| Decision | Choice | Why |
|----------|--------|-----|
| Primary Metric | F1-Score | Balances Precision and Recall |
| Imbalance | scale_pos_weight=559 | Sufficient without sampling |
| Threshold | PR Curve (max F1) | Better than fixed 0.5 |
| Calibration | Isotonic | Reliable probabilities |
| Feature Selection | SelectFromModel (median) | Remove weak features |
| CV Strategy | StratifiedKFold (5) | Preserve fraud ratio |

## Best Model
- **Model:** XGBoost
- **Sampling:** None (scale_pos_weight=559.28)
- **Features:** 16 selected from 31
- **Threshold:** 0.4180
