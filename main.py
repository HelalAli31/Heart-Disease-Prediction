"""
Heart Disease Prediction Using Logistic Regression
==================================================

This script runs the complete machine learning workflow:

    1. Load the dataset (heart.csv)
    2. Exploratory Data Analysis (EDA) + charts saved to results/
    3. Inspect suspicious zero values (Cholesterol, RestingBP)
    4. Preprocessing with a scikit-learn Pipeline + ColumnTransformer
    5. Train a Logistic Regression model on the TRAINING set only
    6. Evaluate the model on the unseen TEST set
    7. Save the trained pipeline to models/heart_disease_model.joblib

Run it from the project folder with:
    python main.py
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Save charts to files without opening windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "heart.csv"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_PATH = MODELS_DIR / "heart_disease_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "HeartDisease"

# Numeric columns: measured values. FastingBS is already 0/1, so it is used as a number.
NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
# Categorical columns: text categories that must be one-hot encoded.
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
# Clinical measurements where a value of 0 is physically impossible for a living patient.
ZERO_MEANS_MISSING = ["RestingBP", "Cholesterol"]


def print_header(title):
    """Print a visible section title in the console."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path):
    """Load heart.csv, or stop with clear download instructions if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {path}\n"
            "Download it from Kaggle:\n"
            "  https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction\n"
            "Unzip the download and place 'heart.csv' in the project folder "
            "(next to main.py), then run this script again."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------
def explore_data(df):
    """Print basic information about the dataset."""
    print_header("EXPLORATORY DATA ANALYSIS")

    print("\nFirst 10 rows:")
    print(df.head(10).to_string())

    print(f"\nDataset dimensions: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumn names and data types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nDescriptive statistics (numeric columns):")
    print(df.describe().round(2).to_string())

    print(f"\nDistribution of {TARGET}:")
    counts = df[TARGET].value_counts().sort_index()
    percents = df[TARGET].value_counts(normalize=True).sort_index() * 100
    for label in counts.index:
        name = "Heart disease" if label == 1 else "No heart disease"
        print(f"  {label} ({name}): {counts[label]} patients ({percents[label]:.1f}%)")


def save_chart(filename):
    """Save the current matplotlib figure to results/ and close it."""
    path = RESULTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path.relative_to(PROJECT_DIR)}")


def plot_eda_charts(df):
    """Create and save the EDA charts."""
    print_header("SAVING EDA CHARTS")
    sns.set_theme(style="whitegrid")
    target_labels = df[TARGET].map({0: "No disease", 1: "Heart disease"})

    # Chart 1: How many patients are in each class? (Checks class balance.)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_labels, order=["No disease", "Heart disease"])
    plt.title("Heart Disease Class Distribution")
    plt.xlabel("")
    plt.ylabel("Number of patients")
    save_chart("01_class_distribution.png")

    # Chart 2: Age distribution, split by class. Shows whether older patients
    # are more often in the heart-disease group.
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="Age", hue=target_labels, bins=20, kde=True)
    plt.title("Age Distribution by Heart Disease Status")
    plt.ylabel("Number of patients")
    save_chart("02_age_distribution.png")

    # Chart 3: Cholesterol distribution. The tall bar at 0 reveals the
    # "missing values recorded as zero" problem handled during cleaning.
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="Cholesterol", hue=target_labels, bins=40)
    plt.title("Cholesterol Distribution (note the spike at 0)")
    plt.ylabel("Number of patients")
    save_chart("03_cholesterol_distribution.png")

    # Chart 4: Heart disease by sex. Compares the proportion of positive
    # cases among male and female patients.
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["Sex"], hue=target_labels)
    plt.title("Heart Disease by Sex")
    plt.xlabel("Sex (M = male, F = female)")
    plt.ylabel("Number of patients")
    save_chart("04_heart_disease_by_sex.png")

    # Chart 5: Correlation heatmap. Shows linear relationships between the
    # numeric columns and the target (values between -1 and +1).
    plt.figure(figsize=(8, 6))
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (numeric columns)")
    save_chart("05_correlation_heatmap.png")

    # Chart 6: Key categorical features vs. the target.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, column in zip(axes, ["ChestPainType", "ExerciseAngina", "ST_Slope"]):
        sns.countplot(x=df[column], hue=target_labels, ax=ax)
        ax.set_title(f"Heart Disease by {column}")
        ax.set_ylabel("Number of patients")
        ax.legend(title="")
    save_chart("06_categorical_features_vs_target.png")


# ---------------------------------------------------------------------------
# 3. Inspect and clean suspicious zero values
# ---------------------------------------------------------------------------
def inspect_zero_values(df):
    """Report how many impossible zero values appear in clinical measurements."""
    print_header("SUSPICIOUS ZERO VALUES")
    for column in ZERO_MEANS_MISSING:
        zero_rows = df[df[column] == 0]
        print(f"\n{column} = 0 in {len(zero_rows)} rows "
              f"({len(zero_rows) / len(df) * 100:.1f}% of the data)")
        if len(zero_rows) > 0:
            disease_rate = zero_rows[TARGET].mean() * 100
            print(f"  Heart disease rate among these rows: {disease_rate:.1f}% "
                  f"(overall rate: {df[TARGET].mean() * 100:.1f}%)")


def clean_data(df):
    """
    Cleaning decisions (documented in README.md):

    * A resting blood pressure or a serum cholesterol of 0 is not a real
      measurement for a living patient. These zeros almost certainly mean
      "value not recorded". We do NOT delete these rows (that would throw
      away many patients and bias the data). Instead we mark them as missing
      (NaN). The pipeline later fills them with the median of the TRAINING
      set and adds a "was missing" indicator column.
    * Replacing 0 with NaN is a row-by-row rule that uses no statistics from
      the data, so doing it before the train/test split causes no leakage.
    * Duplicate rows (if any) are removed so the same patient cannot appear
      in both the training and the test set.
    """
    print_header("DATA CLEANING")
    df = df.copy()

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {duplicates} duplicate rows.")

    for column in ZERO_MEANS_MISSING:
        n_zeros = (df[column] == 0).sum()
        df[column] = df[column].replace(0, np.nan)
        print(f"{column}: {n_zeros} zero values marked as missing (NaN).")

    print(f"Rows kept: {len(df)} (no patients were removed because of zeros).")
    return df


# ---------------------------------------------------------------------------
# 4. Preprocessing + model pipeline
# ---------------------------------------------------------------------------
def build_pipeline():
    """
    Build one Pipeline that contains BOTH preprocessing and the classifier.

    Why a Pipeline? When we call pipeline.fit(X_train, y_train), every step
    (imputer medians, scaler means/std, one-hot categories, model weights)
    learns ONLY from the training data. The test data is only transformed,
    never learned from. This prevents data leakage. The same saved pipeline
    is reused in predict.py, so new patients get identical preprocessing.
    """
    numeric_steps = Pipeline(steps=[
        # Fill missing numbers with the training-set median.
        # add_indicator=True adds a 0/1 column saying "this value was missing".
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        # Rescale each numeric column to mean 0 and standard deviation 1,
        # so features measured on large scales (e.g. Cholesterol) do not
        # dominate features on small scales (e.g. Oldpeak).
        ("scaler", StandardScaler()),
    ])

    categorical_steps = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Turn each category into its own 0/1 column, e.g. Sex -> Sex_M, Sex_F.
        # handle_unknown="ignore" avoids crashes on unseen categories.
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_steps, NUMERIC_FEATURES),
        ("cat", categorical_steps, CATEGORICAL_FEATURES),
    ])

    # -----------------------------------------------------------------------
    # LOGISTIC REGRESSION - how it works
    # -----------------------------------------------------------------------
    # * What it is: a linear model for CLASSIFICATION (despite its name).
    #   It predicts the probability that a sample belongs to class 1.
    #
    # * Why it fits this problem: our target has exactly two classes
    #   (0 = no heart disease, 1 = heart disease), and logistic regression is
    #   the classic, interpretable baseline for binary classification.
    #
    # * Weights and bias: the model gives every feature x_i a weight w_i and
    #   adds a bias b (the "intercept"):
    #         z = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b
    #   A positive weight pushes the prediction towards heart disease,
    #   a negative weight pushes it away. The bias is the starting point
    #   when all (standardized) features are 0, i.e. at their average.
    #
    # * Sigmoid function: z can be any number, so it is squashed into a
    #   probability between 0 and 1:
    #         p = sigmoid(z) = 1 / (1 + e^(-z))
    #   Large positive z -> p close to 1; large negative z -> p close to 0.
    #
    # * Learning from labeled data: during training the model compares its
    #   predicted probabilities with the true labels using "log loss"
    #   (cross-entropy). An optimizer (lbfgs) repeatedly adjusts the weights
    #   and bias to make this loss as small as possible. A small L2
    #   regularization penalty (C=1.0, the default) keeps the weights from
    #   growing too large and helps avoid overfitting.
    #
    # * From probability to class: if p >= 0.5 the model predicts 1
    #   (heart disease), otherwise 0. (The threshold could be lowered to
    #   catch more sick patients, at the cost of more false alarms.)
    # -----------------------------------------------------------------------
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def split_data(df):
    """Separate features/label and make a stratified 80/20 train/test split."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]  # the 11 input features
    y = df[TARGET]                                   # the label

    # stratify=y keeps the same class proportions in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print_header("TRAIN / TEST SPLIT")
    print(f"Training set: {len(X_train)} patients")
    print(f"Test set:     {len(X_test)} patients")
    print(f"Heart disease rate - train: {y_train.mean() * 100:.1f}%, "
          f"test: {y_test.mean() * 100:.1f}%")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
def train_model(pipeline, X_train, y_train):
    """Fit preprocessing + Logistic Regression on the training data only."""
    print_header("TRAINING LOGISTIC REGRESSION")
    pipeline.fit(X_train, y_train)
    print("Model trained on the training set.")
    return pipeline


def show_coefficients(pipeline):
    """Print the learned weights (coefficients) and bias (intercept)."""
    print_header("LEARNED COEFFICIENTS")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    # Make names easier to read: "num__Age" -> "Age", "cat__Sex_M" -> "Sex_M".
    feature_names = [name.split("__", 1)[1] for name in feature_names]

    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0],
        # exp(coefficient) = odds ratio: how much the odds of heart disease
        # are multiplied when this feature increases by one unit
        # (one standard deviation for numeric features, "category present"
        # for one-hot columns), all other features unchanged.
        "odds_ratio": np.exp(model.coef_[0]),
    }).sort_values("coefficient", ascending=False)

    print(coefficients.round(3).to_string(index=False))
    print(f"\nIntercept (bias): {model.intercept_[0]:.3f}")
    print(
        "\nHow to read this:\n"
        "  * coefficient > 0 -> increases the predicted risk of heart disease\n"
        "  * coefficient < 0 -> decreases the predicted risk\n"
        "  * larger absolute value -> stronger influence (numeric features are\n"
        "    standardized, so their coefficients can be compared)\n"
        "  * These are associations learned from this dataset, not proof of cause."
    )
    return coefficients


# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
def evaluate_model(pipeline, X_test, y_test):
    """Evaluate on the unseen test set, print metrics and save charts."""
    print_header("EVALUATION ON THE UNSEEN TEST SET")

    y_pred = pipeline.predict(X_test)                 # class 0 or 1
    y_proba = pipeline.predict_proba(X_test)[:, 1]    # probability of class 1

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(
        y_test, y_pred, target_names=["No disease (0)", "Heart disease (1)"]
    )

    lines = []
    for name, value in metrics.items():
        lines.append(f"{name:<10} {value:.4f}")
    lines += [
        "",
        "Confusion matrix (rows = actual, columns = predicted):",
        f"  True Negatives  (healthy, predicted healthy): {tn}",
        f"  False Positives (healthy, predicted disease): {fp}",
        f"  False Negatives (disease, predicted healthy): {fn}   <-- most dangerous",
        f"  True Positives  (disease, predicted disease): {tp}",
        "",
        "Classification report:",
        report,
        "Metric meanings:",
        "  Accuracy  - share of all test patients classified correctly.",
        "  Precision - of patients predicted 'disease', how many really have it.",
        "  Recall    - of patients who really have disease, how many we caught.",
        "  F1-score  - balance (harmonic mean) of precision and recall.",
        "  ROC-AUC   - how well the probabilities rank sick above healthy",
        "              patients (0.5 = random guessing, 1.0 = perfect).",
        "",
        f"Medical note: the model missed {fn} of {fn + tp} patients who have heart "
        f"disease (false negatives).",
        "A false negative means a sick patient is told they are healthy, which is",
        "worse than a false alarm. That is why Recall is the key metric here.",
    ]
    text = "\n".join(lines)
    print(text)

    # Save metrics to a text file so results can be copied into the README.
    metrics_path = RESULTS_DIR / "metrics.txt"
    metrics_path.write_text(text, encoding="utf-8")
    print(f"\n  Saved: {metrics_path.relative_to(PROJECT_DIR)}")

    plot_confusion_matrix(cm)
    plot_roc_curve(y_test, y_proba, metrics["ROC-AUC"])
    return metrics


def plot_confusion_matrix(cm):
    """Save the confusion matrix as a heatmap."""
    labels = ["No disease (0)", "Heart disease (1)"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion Matrix (test set)")
    save_chart("07_confusion_matrix.png")


def plot_roc_curve(y_test, y_proba, auc):
    """Save the ROC curve (True Positive Rate vs False Positive Rate)."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve (test set)")
    plt.legend(loc="lower right")
    save_chart("08_roc_curve.png")


# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
def save_model(pipeline):
    """Save the full preprocessing + classifier pipeline with joblib."""
    joblib.dump(pipeline, MODEL_PATH)
    print_header("MODEL SAVED")
    print(f"Saved trained pipeline to: {MODEL_PATH.relative_to(PROJECT_DIR)}")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_PATH)
    explore_data(df)
    plot_eda_charts(df)
    inspect_zero_values(df)

    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    pipeline = train_model(build_pipeline(), X_train, y_train)
    show_coefficients(pipeline)
    evaluate_model(pipeline, X_test, y_test)
    save_model(pipeline)


if __name__ == "__main__":
    main()
