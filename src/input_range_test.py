import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/raw/StudentPerformanceFactors.csv"

df = pd.read_csv(DATA_PATH)

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
# TRAIN MODEL ON COMPLETE DATASET
# ============================================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X, y)


# ============================================================
# FUNCTION FOR PREDICTION
# ============================================================

def predict(study_hours, attendance, previous_score):

    input_data = pd.DataFrame({
        "Study_Hours_Daily": [study_hours],
        "Attendance": [attendance],
        "Previous_Scores": [previous_score]
    })

    prediction = model.predict(input_data)[0]

    return prediction


# ============================================================
# TEST 1 — STUDY HOURS 0 TO 24
# ============================================================

print("=" * 75)
print("TEST 1 — STUDY HOURS FROM 0 TO 24")
print("=" * 75)

print(
    f"{'Study Hours':>15} "
    f"{'Attendance':>15} "
    f"{'Previous Score':>18} "
    f"{'Prediction':>15}"
)

print("-" * 75)

for hours in [0, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24]:

    prediction = predict(
        hours,
        85,
        75
    )

    print(
        f"{hours:>15.2f} "
        f"{85:>15.2f} "
        f"{75:>18.2f} "
        f"{prediction:>15.2f}"
    )


# ============================================================
# TEST 2 — ATTENDANCE 0 TO 100
# ============================================================

print("\n" + "=" * 75)
print("TEST 2 — ATTENDANCE FROM 0 TO 100")
print("=" * 75)

print(
    f"{'Study Hours':>15} "
    f"{'Attendance':>15} "
    f"{'Previous Score':>18} "
    f"{'Prediction':>15}"
)

print("-" * 75)

for attendance in [0, 20, 40, 50, 60, 70, 80, 90, 100]:

    prediction = predict(
        5,
        attendance,
        75
    )

    print(
        f"{5:>15.2f} "
        f"{attendance:>15.2f} "
        f"{75:>18.2f} "
        f"{prediction:>15.2f}"
    )


# ============================================================
# TEST 3 — PREVIOUS PERFORMANCE 0 TO 100
# ============================================================

print("\n" + "=" * 75)
print("TEST 3 — PREVIOUS PERFORMANCE FROM 0 TO 100")
print("=" * 75)

print(
    f"{'Study Hours':>15} "
    f"{'Attendance':>15} "
    f"{'Previous Score':>18} "
    f"{'Prediction':>15}"
)

print("-" * 75)

for previous_score in [0, 20, 40, 50, 60, 70, 80, 90, 100]:

    prediction = predict(
        5,
        85,
        previous_score
    )

    print(
        f"{5:>15.2f} "
        f"{85:>15.2f} "
        f"{previous_score:>18.2f} "
        f"{prediction:>15.2f}"
    )


# ============================================================
# TEST 4 — REALISTIC COMBINATIONS
# ============================================================

print("\n" + "=" * 75)
print("TEST 4 — SAMPLE USER INPUTS")
print("=" * 75)

test_cases = [
    (0, 0, 0),
    (2, 50, 50),
    (4, 70, 60),
    (6, 80, 70),
    (8, 90, 80),
    (12, 95, 90),
    (16, 100, 100),
    (24, 100, 100),
]

print(
    f"{'Study':>10} "
    f"{'Attendance':>15} "
    f"{'Previous':>12} "
    f"{'Prediction':>15}"
)

print("-" * 60)

for study, attendance, previous in test_cases:

    prediction = predict(
        study,
        attendance,
        previous
    )

    print(
        f"{study:>10.2f} "
        f"{attendance:>15.2f} "
        f"{previous:>12.2f} "
        f"{prediction:>15.2f}"
    )


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

regressor = model.named_steps["regressor"]

print("\n" + "=" * 75)
print("MODEL COEFFICIENTS")
print("=" * 75)

for feature, coefficient in zip(
    features,
    regressor.coef_
):

    print(
        f"{feature:25s}: {coefficient:.4f}"
    )

print(
    f"\nIntercept: {regressor.intercept_:.4f}"
)


print("\n" + "=" * 75)
print("INPUT RANGE TEST COMPLETED")
print("=" * 75)