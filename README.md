# Heart Disease Prediction Using Logistic Regression

A beginner-friendly supervised machine learning project that predicts whether a
patient has heart disease (yes/no) from 11 clinical features, using
**Logistic Regression** in scikit-learn.

> **Disclaimer:** This is an educational project. Its predictions are **not** a
> medical diagnosis.

---

## 1. Project overview

The project runs the full ML workflow:

1. Load the real dataset (`heart.csv`)
2. Exploratory Data Analysis (EDA): tables and charts
3. Inspect and document data quality problems (impossible zero values)
4. Preprocessing in a scikit-learn `Pipeline` (no data leakage)
5. Train Logistic Regression on 80% of the data
6. Evaluate on the unseen 20% test set
7. Save the trained pipeline and use it to predict a new patient

## 2. Dataset

- **Source:** [Heart Failure Prediction Dataset — fedesoriano, Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size:** 918 patients, 11 input features, 1 target column
- The dataset combines five older heart disease datasets (Cleveland, Hungarian,
  Switzerland, Long Beach VA, Stalog).

| Feature | Meaning | Type |
|---|---|---|
| Age | Age in years | numeric |
| Sex | M = male, F = female | categorical |
| ChestPainType | TA = typical angina, ATA = atypical angina, NAP = non-anginal pain, ASY = asymptomatic | categorical |
| RestingBP | Resting blood pressure (mm Hg) | numeric |
| Cholesterol | Serum cholesterol (mg/dl) | numeric |
| FastingBS | 1 if fasting blood sugar > 120 mg/dl, else 0 | binary (used as numeric) |
| RestingECG | Resting ECG result: Normal, ST, LVH | categorical |
| MaxHR | Maximum heart rate achieved | numeric |
| ExerciseAngina | Exercise-induced angina: Y / N | categorical |
| Oldpeak | ST depression (numeric) | numeric |
| ST_Slope | Slope of peak exercise ST segment: Up, Flat, Down | categorical |

### Target variable

`HeartDisease`: **1** = heart disease, **0** = no heart disease.

### How to get `heart.csv`

1. Sign in to Kaggle and open the dataset link above.
2. Click **Download**, unzip the archive.
3. Copy `heart.csv` into this folder, next to `main.py`.

## 3. Selected algorithm: Logistic Regression

Logistic Regression is a simple, interpretable model for **binary
classification**, so it fits a yes/no target like `HeartDisease`.

**How it works:**

1. **Weights and bias.** Each feature `xᵢ` gets a weight `wᵢ`; a bias `b` is
   added: `z = w₁x₁ + w₂x₂ + … + wₙxₙ + b`.
   A positive weight raises the risk, a negative weight lowers it.
2. **Sigmoid function.** `z` can be any number, so it is converted into a
   probability between 0 and 1: `p = 1 / (1 + e^(−z))`.
3. **Learning.** During training the model compares its probabilities with the
   true labels using *log loss* (cross-entropy), and an optimizer adjusts the
   weights and bias to make that loss as small as possible.
4. **Decision.** If `p ≥ 0.5` the model predicts **1** (heart disease),
   otherwise **0**.

Because the weights are visible, we can explain *which* features push a
prediction towards heart disease, which is useful in a medical setting.

## 4. Data preprocessing and cleaning decisions

| Step | What and why |
|---|---|
| Split X / y | The 11 features are `X`; `HeartDisease` is `y`. |
| Train/test split | 80% train / 20% test, `stratify=y` (same class ratio in both), `random_state=42`. |
| Duplicates | Checked; any duplicates are removed so a patient can't appear in both train and test. |
| **Zero values** | `Cholesterol = 0` and `RestingBP = 0` are impossible for a living patient, so they almost certainly mean "not recorded". They are **converted to missing (NaN)**, **not deleted**. Removing them would throw away a large share of patients. `main.py` prints how many rows are affected. |
| Missing values | Numeric: filled with the **training-set median**, plus a 0/1 "was missing" indicator column. Categorical: most frequent value. |
| Numeric scaling | `StandardScaler` → mean 0, standard deviation 1. |
| Categorical encoding | `OneHotEncoder` → one 0/1 column per category. |
| Leakage prevention | All of these steps live inside one `Pipeline` + `ColumnTransformer`, so they are fitted **only on training data**. The test set is only transformed. |

## 5. Project structure

```
heart-disease-prediction/
├── heart.csv            # dataset (download from Kaggle)
├── main.py              # EDA, preprocessing, training, evaluation, saving
├── predict.py           # loads the saved model and predicts one new patient
├── requirements.txt     # Python dependencies
├── README.md
├── models/              # heart_disease_model.joblib (created by main.py)
└── results/             # charts and metrics.txt (created by main.py)
```

## 6. Installation (Windows / VS Code)

1. Install **Python 3.10+** from <https://www.python.org/downloads/>
   (tick **"Add python.exe to PATH"** during installation).
2. Open this folder in VS Code and open a terminal (**Terminal → New Terminal**).
3. Create and activate a virtual environment, then install the packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `python` is "not found" right after installing it, close and reopen VS Code
so the terminal picks up the new PATH. If PowerShell blocks `Activate.ps1`,
run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## 7. How to run training

```powershell
python main.py
```

This prints the EDA, cleaning steps, learned coefficients and evaluation
metrics, and creates:

| File | What it shows |
|---|---|
| `results/01_class_distribution.png` | How many patients are in each class (checks class balance). |
| `results/02_age_distribution.png` | Age histogram split by class: heart disease is more common in older patients. |
| `results/03_cholesterol_distribution.png` | Cholesterol histogram; the spike at 0 reveals the missing-value problem. |
| `results/04_heart_disease_by_sex.png` | Number of positive/negative cases for male and female patients. |
| `results/05_correlation_heatmap.png` | Linear correlation (−1 to +1) between numeric columns and the target. |
| `results/06_categorical_features_vs_target.png` | Heart disease by ChestPainType, ExerciseAngina and ST_Slope. |
| `results/07_confusion_matrix.png` | Correct and wrong predictions on the test set. |
| `results/08_roc_curve.png` | Trade-off between catching sick patients and false alarms at every threshold. |
| `results/metrics.txt` | All evaluation numbers from the latest run. |
| `models/heart_disease_model.joblib` | The trained preprocessing + model pipeline. |

## 8. How to predict a new patient

```powershell
python predict.py
```

The script loads the saved pipeline, applies the same preprocessing
automatically, and prints the predicted class (0/1) and the probability of
heart disease. To test another patient, edit the `EXAMPLE_PATIENT` dictionary
in `predict.py` (all 11 features are required).

## 9. Evaluation metrics explained

| Metric | Simple meaning |
|---|---|
| **Accuracy** | Percentage of all test patients classified correctly. |
| **Precision** | Of the patients the model flagged as "heart disease", how many really have it. |
| **Recall** (sensitivity) | Of the patients who really have heart disease, how many the model caught. |
| **F1-score** | One number balancing precision and recall (their harmonic mean). |
| **Confusion matrix** | Table of true negatives, false positives, false negatives, true positives. |
| **ROC curve** | True positive rate vs. false positive rate for every possible threshold. |
| **ROC-AUC** | Area under the ROC curve: 0.5 = random guessing, 1.0 = perfect ranking. |

**Why recall matters most here:** a **false negative** is a sick patient the
model calls healthy. In a medical context, that is far more harmful than a
false alarm (false positive), which would just lead to more tests. So recall
on class 1 is the key metric to look at.

## 10. Results

Actual output of `python main.py` (Python 3.12, scikit-learn 1.9.1,
`random_state=42`). The test set is 184 patients the model never saw during
training. The full output is in `results/metrics.txt`.

| Metric | Value |
|---|---|
| Accuracy | 0.891 (164 of 184 correct) |
| Precision | 0.887 |
| Recall | 0.922 (94 of 102 sick patients caught) |
| F1-score | 0.904 |
| ROC-AUC | 0.932 |

**Confusion matrix (test set):**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** (no disease) | 70 (TN) | 12 (FP) |
| **Actual 1** (disease) | **8 (FN)** | 94 (TP) |

The model missed **8 of 102** patients with heart disease (false negatives).

**What the data and model show:**
- 172 of 918 patients (18.7%) have `Cholesterol = 0`, and 88.4% of them have
  heart disease. One patient has `RestingBP = 0`.
- Features with the largest positive coefficients (raise predicted risk):
  `ChestPainType_ASY`, `ST_Slope_Flat`, the Cholesterol "was missing"
  indicator, `Sex_M`, `ExerciseAngina_Y`.
- The largest negative coefficient (lowers predicted risk) is `ST_Slope_Up`.
- The example patient in `predict.py` is predicted class **1**, with a
  97.4% probability of heart disease.

## 11. Limitations

- **Small dataset:** 918 patients; one 184-patient test set gives only a
  rough estimate of real-world performance.
- **Missing values recorded as 0:** many Cholesterol values are 0 (not
  recorded). Imputation is a guess. These zeros are concentrated in some
  source hospitals, so the "was missing" indicator can partly learn *which
  hospital* a patient came from rather than real medical risk.
- **Combined sources:** data from five hospitals with different populations
  and measurement practices, collected decades ago.
- **Imbalanced demographics:** most patients are male, so the model may be
  less reliable for female patients.
- **Correlation, not causation:** coefficients show associations in this
  data, not medical causes.
- **Fixed 0.5 threshold:** lowering it would raise recall (fewer missed
  patients) at the cost of more false alarms.
- **Not a medical tool:** not clinically validated; for education only.
#   H e a r t - D i s e a s e - P r e d i c t i o n  
 