import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/raw/StudentPerformanceFactors.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# CLEAN TARGET
# ============================================================

# Exam scores should be on a 0–100 scale.
# The dataset contains one 101 value.
# Treat it as the documented upper-bound outlier.

df["Exam_Score"] = df["Exam_Score"].clip(0, 100)


# ============================================================
# FEATURES
# ============================================================

df["Study_Hours_Daily"] = df["Hours_Studied"] / 7

features = [
    "Study_Hours_Daily",
    "Attendance",
    "Previous_Scores"
]

target = "Exam_Score"

X = df[features]
y = df[target]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# STANDARD LINEAR REGRESSION
# ============================================================

linear_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)


# ============================================================
# BOUNDED LINEAR MODEL
# ============================================================

bounded_linear_pred = np.clip(
    linear_pred,
    0,
    100
)


# ============================================================
# LOGISTIC BOUNDED REGRESSION
# ============================================================

class BoundedRegression:

    def __init__(self):
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])

    def _logit(self, p):
        p = np.clip(p, 0.001, 0.999)
        return np.log(p / (1 - p))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        # Convert score to 0–1
        p = np.clip(y / 100.0, 0.001, 0.999)

        # Convert 0–1 target to log-odds
        transformed_y = self._logit(p)

        self.model.fit(X, transformed_y)

        return self

    def predict(self, X):

        transformed_prediction = self.model.predict(X)

        probability = self._sigmoid(
            transformed_prediction
        )

        prediction = probability * 100

        return np.clip(prediction, 0, 100)


bounded_model = BoundedRegression()

bounded_model.fit(
    X_train,
    y_train
)

bounded_pred = bounded_model.predict(
    X_test
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(name, actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print(
        f"{name:<30}"
        f"MAE = {mae:.4f}    "
        f"RMSE = {rmse:.4f}    "
        f"R² = {r2:.4f}"
    )

    return mae, rmse, r2


# ============================================================
# MODEL COMPARISON
# ============================================================

print("=" * 85)
print("BOUNDED MODEL COMPARISON")
print("=" * 85)

print(f"Training students : {len(X_train)}")
print(f"Test students     : {len(X_test)}")

print("\n")

evaluate(
    "Linear Regression",
    y_test,
    linear_pred
)

evaluate(
    "Linear + Bounding",
    y_test,
    bounded_linear_pred
)

evaluate(
    "Logistic Bounded Regression",
    y_test,
    bounded_pred
)


# ============================================================
# FULL INPUT RANGE TEST
# ============================================================

print("\n" + "=" * 85)
print("FULL INPUT RANGE TEST")
print("=" * 85)

print(
    f"{'Study':>8}"
    f"{'Attendance':>15}"
    f"{'Previous':>12}"
    f"{'Linear':>15}"
    f"{'Bounded Linear':>18}"
    f"{'Bounded Model':>18}"
)

print("-" * 85)


test_inputs = [
    (0, 0, 0),
    (2, 50, 50),
    (4, 70, 60),
    (6, 80, 70),
    (8, 90, 80),
    (12, 95, 90),
    (16, 100, 100),
    (20, 100, 100),
    (24, 100, 100)
]


test_df = pd.DataFrame(
    [
        {
            "Study_Hours_Daily": study,
            "Attendance": attendance,
            "Previous_Scores": previous
        }
        for study, attendance, previous
        in test_inputs
    ]
)


linear_range_pred = linear_model.predict(
    test_df
)

bounded_linear_range_pred = np.clip(
    linear_range_pred,
    0,
    100
)

bounded_range_pred = bounded_model.predict(
    test_df
)


for i, values in enumerate(test_inputs):

    study, attendance, previous = values

    print(
        f"{study:>8.2f}"
        f"{attendance:>15.2f}"
        f"{previous:>12.2f}"
        f"{linear_range_pred[i]:>15.2f}"
        f"{bounded_linear_range_pred[i]:>18.2f}"
        f"{bounded_range_pred[i]:>18.2f}"
    )


# ============================================================
# RANGE CHECK
# ============================================================

print("\n" + "=" * 85)
print("BOUND CHECK")
print("=" * 85)

print(
    f"Bounded model minimum : "
    f"{bounded_range_pred.min():.2f}"
)

print(
    f"Bounded model maximum : "
    f"{bounded_range_pred.max():.2f}"
)

if (
    bounded_range_pred.min() >= 0
    and bounded_range_pred.max() <= 100
):

    print("\nPASS: All bounded predictions are within 0–100.")

else:

    print("\nFAIL: Prediction outside 0–100.")


print("\n" + "=" * 85)
print("TEST COMPLETED")
print("=" * 85)