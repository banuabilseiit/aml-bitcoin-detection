# 🔍 AML Monitoring of Cryptocurrency Transactions
### RegTech | Machine Learning | Financial Supervision

Automated detection of suspicious Bitcoin transactions using machine learning — built from a financial regulator's perspective.

---

## 📋 Project Overview

**Business Problem:** Manual review of hundreds of thousands of cryptocurrency transactions is impossible. Fraudsters use Bitcoin for money laundering, ransomware, darknet markets, and terrorist financing.

**ML Solution:** A binary classification model that assigns a risk score to each transaction and flags suspicious ones for compliance officer review — automating AML screening in line with FATF Recommendation 16 and AIFC standards.

---

## 📦 Dataset

**Source:** [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — Kaggle  
**Created by:** Elliptic + MIT-IBM Watson AI Lab (presented at KDD 2019)

| File | Size | Description |
|---|---|---|
| elliptic_txs_features.csv | 203,769 rows, 167 cols | Transaction features: txId, timestep, 94 local + 72 aggregated features |
| elliptic_txs_classes.csv | 203,769 rows, 2 cols | Class labels: 1 (illicit), 2 (licit), unknown |
| elliptic_txs_edgelist.csv | 234,355 rows, 2 cols | Transaction graph edges — not used in this project |

**Key facts:**
- Total transaction value: **$6 billion**
- 49 time steps (~2 weeks each, ~2 years total)
- Class distribution: 2.2% illicit, 20.6% licit, 77.2% unknown
- No missing values

---

## 📓 Notebook Structure

| Section | Description |
|---|---|
| 1. Data Loading | Download dataset via kagglehub, load features and classes |
| 2. Data Quality | Check missing values |
| 3. Feature Preparation | Rename columns, merge datasets |
| 4. EDA | Class distribution, temporal dynamics, correlation analysis |
| 5. Hypothesis Testing | T-test, Spearman, KS-test |
| 6. Preprocessing | SMOTE balancing, train/test split |
| 7. Model Training | Logistic Regression, Random Forest, XGBoost |
| 8. Model Evaluation | Confusion Matrix, ROC curves, comparison table, cross-validation |
| 9. Hyperparameter Tuning | RandomizedSearchCV for XGBoost |
| 10. SHAP Interpretability | Feature importance and impact analysis |
| 11. Data Drift | PSI monitoring across time periods |
| 12. Model Saving | Save model and dependencies |

---

## 🔬 Hypothesis Testing

| Hypothesis | Test | Result |
|---|---|---|
| Illicit transactions have anomalous parameters | T-test | ✅ Confirmed (p < 0.0001, t=58.45) |
| Fraud rate increases over time | Spearman | ❌ Rejected (r=-0.009, p=0.95) |
| Class distributions differ significantly | KS-test | ✅ Confirmed (KS=0.617, p < 0.0001) |

---

## 📊 Results

| Model | ROC-AUC | F1 (illicit) | Recall | Precision |
|---|---|---|---|---|
| Logistic Regression | 0.9726 | 0.72 | 0.88 | 0.61 |
| Random Forest | 0.9971 | 0.94 | 0.90 | 0.99 |
| **XGBoost** | **0.9977** | **0.96** | **0.93** | **0.99** |

**XGBoost** detected **845 out of 909** illicit transactions with only **8 false alarms**.

### Cross-Validation (5-fold)

| Model | ROC-AUC | Std |
|---|---|---|
| Logistic Regression | 0.9499 | ±0.027 |
| Random Forest | 0.9718 | ±0.036 |
| XGBoost | 0.9661 | ±0.031 |

No overfitting detected.

---

## 🧠 SHAP Interpretability

- **local_feat_5** — most influential; low values strongly indicate illicit activity
- **local_feat_53**, **local_feat_59** — high values increase fraud probability
- 14 out of 15 top features are **local** — fraud is primarily identified by a transaction's own parameters

---

## 📉 Data Drift (PSI)

| Period | PSI | Status |
|---|---|---|
| 11–20 | 0.36 | 🚨 Critical drift |
| 21–30 | 0.39 | 🚨 Critical drift |
| 31–40 | 0.98 | 🚨 Critical drift |
| 41–49 | 0.86 | 🚨 Critical drift |

Critical drift across all periods — model retraining recommended at least monthly.

---

## 🏛️ RegTech Value

```
Bitcoin Blockchain
       ↓
   ML Model (XGBoost)
       ↓
  Risk Score (0–100%)
       ↓
> 30% risk → Compliance Officer → Decision
< 30% risk → Auto-cleared
```

Complies with FATF Recommendation 16 and AIFC regulatory standards.

---

## 🚀 How to Run

### Local
```bash
git clone https://github.com/your-username/aml-bitcoin-detection
cd aml-bitcoin-detection
pip install -r requirements.txt
jupyter notebook AML_final.ipynb
```

### Docker
```bash
docker build -t aml-detection .
docker run -p 8888:8888 aml-detection
# Open http://localhost:8888
```

---

## 📁 Project Structure

```
├── AML_final.ipynb       # Main notebook
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker setup
└── README.md             # This file
```

---

*ML Project — RegTech | AML Monitoring of Digital Assets*
