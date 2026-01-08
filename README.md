<div align="center">

# 🛡️ Fraud Detection System

### *Catching Financial Criminals with Machine Learning*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![XGBoost](https://img.shields.io/badge/XGBoost-Powered-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

<img src="https://readme-typing-svg. herokuapp.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00D4FF&center=true&vCenter=true&random=false&width=600&lines=6.3M%2B+Transactions+Analyzed;99%25+AUPRC+Score+Achieved;Real-Time+Fraud+Detection;XGBoost+ML+Pipeline" alt="Typing SVG" />

<br>

**An end-to-end machine learning solution for detecting fraudulent financial transactions with exceptional accuracy**

[🚀 Quick Start](#-quick-start) •
[📊 Results](#-model-performance) •
[🔬 Methodology](#-methodology) •
[📁 Dataset](#-dataset)

---

</div>

## 🎯 Project Highlights

<table>
<tr>
<td width="50%">

### 🔢 By The Numbers

| Metric | Value |
|--------|-------|
| **Transactions Processed** | 6,362,620 |
| **Fraud Cases Detected** | 8,213 |
| **AUPRC Score** | 0.9903 |
| **Precision (Fraud)** | 88% |
| **Recall (Fraud)** | 100% |
| **Overall Accuracy** | 99.7%+ |

</td>
<td width="50%">

### ⚡ Key Features

- 🧠 **XGBoost Classifier** with optimized hyperparameters
- ⚖️ **Imbalanced Data Handling** via scale_pos_weight
- 🔧 **Custom Feature Engineering** (Error Balance Detection)
- 📈 **Early Stopping** to prevent overfitting
- 🎨 **Rich Visualizations** for model interpretation
- 🏗️ **Object-Oriented Design** for clean, reusable code

</td>
</tr>
</table>

---

## 📖 Overview

Financial fraud poses a **$5+ trillion** global threat annually. This project implements a **production-ready fraud detection system** capable of analyzing millions of transactions and identifying fraudulent patterns with **near-perfect recall**. 

The system focuses on **TRANSFER** and **CASH_OUT** transactions — the two types where 100% of fraudulent activities occur in the dataset — ensuring computational efficiency while maximizing detection capability.

```
┌─────────────────────────────────────────────────────────────────┐
│                    🔄 DETECTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📥 Data Loading    →    🔍 EDA & Analysis                    │
│         ↓                       ↓                               │
│   🛠️ Feature Engineering  →  ⚙️ Model Training                 │
│         ↓                       ↓                               │
│   📊 Evaluation      →    🎯 Fraud Detection                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required packages
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Siddharthk17/Fraud-Detection. git
cd Fraud-Detection

# Download the dataset from Kaggle
# https://www.kaggle.com/datasets/sid17a/fraud-detection-dataset
```

### Run Detection

```python
from fraud_detection import FraudDetector

# Initialize and run the complete pipeline
detector = FraudDetector("path/to/Fraud.csv")
detector.load_data()
detector.perform_eda()
detector.preprocess_and_feature_engineering()
detector.split_data()
detector.train_model()
detector.evaluate_model()
```

---

## 📊 Model Performance

<div align="center">

### 🏆 Classification Report

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    552,439
           1       0.88      1.00      0.94      1,643

    accuracy                           1.00    554,082
   macro avg       0.94      1.00      0.97    554,082
weighted avg       1.00      1.00      1.00    554,082
```

</div>

### 📈 Performance Visualizations

| Confusion Matrix | Feature Importance | Precision-Recall Curve |
|: ----------------:|:------------------:|:----------------------:|
| Perfect separation of legitimate vs fraudulent transactions | 
`errorBalanceOrig` and `errorBalanceDest` are top predictors | 
AUPRC = 0.9903 demonstrates excellent model calibration |

---

## 🔬 Methodology

### 1️⃣ Exploratory Data Analysis

```python
# Key insight:  Fraud ONLY occurs in TRANSFER and CASH_OUT
type_group = df.groupby('type')['isFraud'].sum()

# Results: 
# CASH_IN     →     0 fraud cases
# CASH_OUT    → 4,116 fraud cases  ✓
# DEBIT       →     0 fraud cases
# PAYMENT     →     0 fraud cases
# TRANSFER    → 4,097 fraud cases  ✓
```

### 2️⃣ Feature Engineering

Two powerful engineered features that expose fraudster behavior:

```python
# Error Balance Origin - Detects balance manipulation at source
errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg

# Error Balance Destination - Detects balance manipulation at destination
errorBalanceDest = oldbalanceDest + amount - newbalanceDest
```

> 💡 **Why this works:** Fraudsters often manipulate the system so that balance changes don't match transaction amounts. These features capture that discrepancy.

### 3️⃣ Handling Class Imbalance

With only **0.129% fraud rate**, we employ: 

- **Stratified Train-Test Split** — Maintains fraud ratio in both sets
- **scale_pos_weight** — Automatically balances class weights in XGBoost
- **AUPRC Metric** — Superior to ROC-AUC for imbalanced datasets

### 4️⃣ Model Architecture

```python
XGBClassifier(
    n_estimators=1000,          # Maximum trees
    max_depth=6,                # Prevent overfitting
    learning_rate=0.05,         # Gradual learning
    subsample=0.8,              # Row sampling
    colsample_bytree=0.8,       # Feature sampling
    scale_pos_weight=weights,   # Class balancing
    tree_method='hist',         # Fast training
    eval_metric='aucpr',        # Optimized for imbalance
    early_stopping_rounds=50    # Prevent overfitting
)
```

---

## 📁 Dataset

<div align="center">

📊 **[Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/sid17a/fraud-detection-dataset)**

</div>

### Dataset Features

| Feature | Description | Type |
|---------|-------------|------|
| `step` | Time step (1 step = 1 hour) | Integer |
| `type` | Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN) | Categorical |
| `amount` | Transaction amount | Float |
| `nameOrig` | Sender ID (anonymized) | String |
| `oldbalanceOrg` | Sender's balance before transaction | Float |
| `newbalanceOrig` | Sender's balance after transaction | Float |
| `nameDest` | Receiver ID (anonymized) | String |
| `oldbalanceDest` | Receiver's balance before transaction | Float |
| `newbalanceDest` | Receiver's balance after transaction | Float |
| `isFraud` | **Target variable** (1 = Fraud, 0 = Legitimate) | Binary |
| `isFlaggedFraud` | System-flagged as suspicious | Binary |

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## 📂 Project Structure

```
Fraud-Detection/
│
├── 📓 Fraud Detection. ipynb    # Main notebook with complete pipeline
├── 📄 README.md                # You are here! 
└── 📊 Fraud. csv                # Dataset (download from Kaggle)
```

---

## 🔮 Future Improvements

- [ ] 🌐 Deploy as REST API using FastAPI/Flask
- [ ] 📱 Build real-time streaming detection with Apache Kafka
- [ ] 🧪 Experiment with deep learning (LSTM for temporal patterns)
- [ ] 🔄 Implement SMOTE/ADASYN for synthetic minority oversampling
- [ ] 📊 Add SHAP values for model explainability
- [ ] 🐳 Containerize with Docker for easy deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repository
2. 🔧 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🔀 Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

<div align="center">

**Siddharth K**

![GitHub](https://img.shields.io/badge/GitHub-Siddharthk17-181717?style=for-the-badge&logo=github)
---

<img src="https://readme-typing-svg. herokuapp.com?font=Fira+Code&weight=500&size=18&pause=1000&color=58A6FF&center=true&vCenter=true&random=false&width=500&lines=Thanks+for+visiting! +%E2%AD%90;Star+this+repo+if+you+found+it+useful!" alt="Footer" />

</div>
