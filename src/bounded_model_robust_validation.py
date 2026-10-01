import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# BOUNDED MODEL
# ============================================================

class BoundedRegression:
    """
    Regression model whose final prediction is naturally bounded
    between 0 and 100 using a logistic transformation.
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        # Convert 0-100 target to 0-1
        y_scaled = np.clip(y / 100.0, 0.001, 0.999)

        # Logit transformation
        y_logit = np.log(y_scaled / (1 - y_scaled))

        self.model.fit(X, y_logit)

        return self

    def predict(self, X):
        # Linear prediction in logit space
        z = self.model.predict(X)

        # Sigmoid
        probability = 1 / (1 + np.exp(-z))

        # Convert back to 0-100
        prediction = probability * 100

        # Final safety bound
        return np.clip(prediction, 0, 100)


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/raw/StudentPerformanceFactors.csv"

df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

# Exam score should be on 0-100 scale.
# Dataset contains one 101 value, so clip it to 100.
df["Exam_Score"] = df["Exam_Score"].clip(0, 100)

# ------------------------------------------------------------
# Convert weekly study hours to daily hours
# ------------------------------------------------------------

df["Study_Hours_Daily"] = df["Hours_Studied"] / 7.0


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Study_Hours_Daily",
    "Attendance",
    "Previous_Scores"
]

X = df[FEATURES]
y = df["Exam_Score"]


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)


results = {
    "Linear Regression": {
        "MAE": [],
        "RMSE": [],
        "R2": []
    },

    "Bounded Linear Regression": {
        "MAE": [],
        "RMSE": [],
        "R2": []
    },

    "Logistic Bounded Regression": {
        "MAE": [],
        "RMSE": [],
        "R2": []
    }
}


# ============================================================
# RUN VALIDATION
# ============================================================

for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    # --------------------------------------------------------
    # 1. Standard Linear Regression
    # --------------------------------------------------------

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    pred_linear = linear_model.predict(X_test)


    # --------------------------------------------------------
    # 2. Linear Regression + Bounding
    # --------------------------------------------------------

    pred_bounded_linear = np.clip(
        pred_linear,
        0,
        100
    )


    # --------------------------------------------------------
    # 3. Logistic Bounded Regression
    # --------------------------------------------------------

    bounded_model = BoundedRegression()
    bounded_model.fit(X_train, y_train)

    pred_bounded = bounded_model.predict(X_test)


    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    predictions = {
        "Linear Regression": pred_linear,
        "Bounded Linear Regression": pred_bounded_linear,
        "Logistic Bounded Regression": pred_bounded
    }


    for model_name, predictions_array in predictions.items():

        mae = mean_absolute_error(
            y_test,
            predictions_array
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions_array
            )
        )

        r2 = r2_score(
            y_test,
            predictions_array
        )

        results[model_name]["MAE"].append(mae)
        results[model_name]["RMSE"].append(rmse)
        results[model_name]["R2"].append(r2)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 80)
print("ROBUST BOUNDED MODEL VALIDATION")
print("=" * 80)

print()
print("Dataset students :", len(df))
print("Features          :", ", ".join(FEATURES))
print("Validation        : 5-Fold × 5-Repeated")
print()


for model_name, metrics in results.items():

    print("-" * 80)
    print(model_name)

    print(
        f"MAE  : {np.mean(metrics['MAE']):.4f}"
        f" ± {np.std(metrics['MAE']):.4f}"
    )

    print(
        f"RMSE : {np.mean(metrics['RMSE']):.4f}"
        f" ± {np.std(metrics['RMSE']):.4f}"
    )

    print(
        f"R²   : {np.mean(metrics['R2']):.4f}"
        f" ± {np.std(metrics['R2']):.4f}"
    )


# ============================================================
# RANGE CHECK
# ============================================================

print()
print("=" * 80)
print("FINAL RANGE CHECK")
print("=" * 80)

# Train bounded model on complete dataset
final_model = BoundedRegression()
final_model.fit(X, y)

stress_tests = np.array([
    [0, 0, 0],
    [2, 50, 50],
    [4, 70, 60],
    [6, 80, 70],
    [8, 90, 80],
    [12, 95, 90],
    [16, 100, 100],
    [20, 100, 100],
    [24, 100, 100]
])

stress_predictions = final_model.predict(stress_tests)

print()
print(
    f"{'Study':>8}"
    f"{'Attendance':>15}"
    f"{'Previous':>12}"
    f"{'Prediction':>15}"
)

print("-" * 55)

for row, prediction in zip(stress_tests, stress_predictions):

    print(
        f"{row[0]:>8.2f}"
        f"{row[1]:>15.2f}"
        f"{row[2]:>12.2f}"
        f"{prediction:>15.2f}"
    )


print()
print("Minimum prediction :", f"{stress_predictions.min():.2f}")
print("Maximum prediction :", f"{stress_predictions.max():.2f}")

if np.all(
    (stress_predictions >= 0) &
    (stress_predictions <= 100)
):
    print()
    print("PASS: All predictions remain within 0-100.")
else:
    print()
    print("FAIL: Prediction outside valid range detected.")

print()
print("=" * 80)
print("VALIDATION COMPLETED")
print("=" * 80)