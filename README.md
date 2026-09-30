# 🚢 Titanic Survival Predictor (Machine Learning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sumant26/titanic-survival-ml/blob/main/titanic_survival.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic%20Competition-20BEFF.svg)](https://www.kaggle.com/competitions/titanic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning classification project predicting passenger survival on the Titanic (1912 disaster) using demographic and ticket information.

---

## 📌 Project Overview

The goal of this project is to build and evaluate binary classification models to predict whether a passenger survived (`1`) or died (`0`). 

Unlike regression problems that predict continuous numeric outcomes, this classification task focuses on probabilistic decision boundaries, dealing with class imbalances, evaluation trade-offs (Precision vs. Recall), and domain-specific feature engineering.

---

## 🛠️ Key Highlights & Skills

- **Classification Algorithms:** Logistic Regression, Random Forest, HistGradientBoosting / Gradient Boosting.
- **Feature Engineering:** Extracted passenger titles (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`) from names, derived family size metrics (`FamilySize`, `IsAlone`), and cabin presence indicators (`HasCabin`).
- **Data Preprocessing Pipelines:** Modular `scikit-learn` `ColumnTransformer` handling numerical imputation (`median` + `StandardScaler`) and categorical encoding (`most_frequent` + `OneHotEncoder`).
- **Comprehensive Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and Classification Probability Threshold Tuning.
- **Validation & Submission:** 5-fold Stratified Cross-Validation and Kaggle leaderboard submission workflow.

---

## 📊 Dataset & Features

The dataset is sourced from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

| Feature | Description | Type / Values |
|---|---|---|
| `Survived` | **Target Variable** | `0` = Died, `1` = Survived |
| `Pclass` | Ticket Class | `1` = 1st (Upper), `2` = 2nd (Middle), `3` = 3rd (Lower) |
| `Name` | Passenger Name | Text (used to extract `Title`) |
| `Sex` | Gender | `male`, `female` |
| `Age` | Age in years | Continuous (imputed with median) |
| `SibSp` | Number of Siblings / Spouses aboard | Discrete integer |
| `Parch` | Number of Parents / Children aboard | Discrete integer |
| `Ticket` | Ticket number | Alphanumeric string |
| `Fare` | Passenger fare | Continuous |
| `Cabin` | Cabin number | String (mostly missing, mapped to `HasCabin`) |
| `Embarked` | Port of Embarkation | `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton |

### Engineered Features
- **`Title`**: Extracted title from `Name` (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`).
- **`FamilySize`**: `SibSp + Parch + 1`
- **`IsAlone`**: Binary indicator (`1` if `FamilySize == 1`, else `0`).
- **`HasCabin`**: Binary indicator (`1` if `Cabin` is recorded, else `0`).

---

## 📈 Model Performance & Expected Results

| Model | Cross-Validation / Test Accuracy | Notes |
|---|---|---|
| **Majority Baseline** | ~61.6% | Predicts all passengers died (`0`) |
| **Logistic Regression** | ~80.0% – 83.0% | Linear decision boundary with scaled features |
| **Random Forest Classifier** | ~81.0% – 83.5% | Non-linear tree ensemble (`n_estimators=300, max_depth=6`) |
| **Gradient Boosting (`HistGradientBoosting`)** | ~81.5% – 84.0% | Boosting trees with learning rate calibration |
| **Kaggle Leaderboard Benchmark** | **~77.0% – 79.5%** | First submission on unseen Kaggle test set |

---

## 🚀 Getting Started

### Option 1: Run in Google Colab (Recommended)
1. Click the **[Open In Colab](https://colab.research.google.com/github/Sumant26/titanic-survival-ml/blob/main/titanic_survival.ipynb)** badge at the top.
2. Run each cell sequentially (**Shift + Enter**). The dataset is downloaded automatically.

### Option 2: Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sumant26/titanic-survival-ml.git
   cd titanic-survival-ml
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```

4. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook titanic_survival.ipynb
   ```

---

## 🏆 Kaggle Submission Guide

To generate predictions for the Kaggle competition leaderboard:
1. Download `train.csv` and `test.csv` from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data).
2. Place both CSV files in the project root directory (or upload them to Colab).
3. Run Section 13 in `titanic_survival.ipynb` to generate `submission.csv`.
4. Submit `submission.csv` via the Kaggle competition submission page.

---

## 📂 Project Structure

```text
titanic-survival-ml/
├── README.md               # Project documentation & overview
└── titanic_survival.ipynb  # Main Jupyter notebook with EDA, modeling, and evaluation
```

---

## 💡 Key Learnings

1. **Accuracy Paradox:** High accuracy alone can be misleading when classes are imbalanced; confusion matrices and precision/recall curves provide true insight into classifier behavior.
2. **Threshold Tuning:** Shifting probability thresholds allows tuning between high precision (avoiding false alarms) and high recall (catching all positive cases).
3. **Feature Engineering Impact:** Domain-relevant feature derivation (such as passenger social titles and family units) significantly boosts model signal compared to raw features.

---

## 👤 Author

**Sumant Tulshibagwale**
- GitHub: [@Sumant26](https://github.com/Sumant26)
