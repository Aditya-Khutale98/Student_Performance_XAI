import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/raw/StudentPerformanceFactors.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. PREPARE OUR THREE FEATURES
# ============================================================

# Dataset stores study hours per week.
# Convert to daily equivalent.

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
# 3. TRAIN / FINAL TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 4. MODEL PIPELINE
# ============================================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "regressor",
        LinearRegression()
    )
])


# ============================================================
# 5. CROSS VALIDATION ON TRAINING DATA ONLY
# ============================================================

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

cv_scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    },
    n_jobs=-1
)

cv_mae = -cv_scores["test_MAE"]
cv_rmse = -cv_scores["test_RMSE"]
cv_r2 = cv_scores["test_R2"]


# ============================================================
# 6. TRAIN ON 80% TRAINING DATA
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 7. PREDICT UNSEEN 20%
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. FINAL TEST METRICS
# ============================================================

test_mae = mean_absolute_error(
    y_test,
    y_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

test_r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("=" * 75)
print("FINAL MODEL EVALUATION")
print("=" * 75)

print("\nDATA")
print("-" * 75)

print(f"Total students       : {len(df)}")
print(f"Training students    : {len(X_train)}")
print(f"Unseen test students : {len(X_test)}")

print("\nFEATURES")
print("-" * 75)

for feature in features:
    print(f"- {feature}")

print(f"\nTarget: {target}")


# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("REPEATED CROSS-VALIDATION ON TRAINING DATA")
print("=" * 75)

print(f"MAE  : {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
print(f"RMSE : {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
print(f"R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")


# ============================================================
# FINAL UNSEEN TEST
# ============================================================

print("\n" + "=" * 75)
print("FINAL UNSEEN TEST RESULTS")
print("=" * 75)

print(f"MAE  : {test_mae:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"R²   : {test_r2:.4f}")


# ============================================================
# ERROR INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("ERROR INTERPRETATION")
print("=" * 75)

print(
    f"On average, predictions differ from the actual "
    f"exam score by about {test_mae:.2f} score points."
)

print(
    f"Typical larger prediction error "
    f"(RMSE): {test_rmse:.2f} score points."
)


# ============================================================
# TARGET RANGE
# ============================================================

print("\n" + "=" * 75)
print("TARGET RANGE")
print("=" * 75)

print(f"Actual minimum : {y.min():.2f}")
print(f"Actual maximum : {y.max():.2f}")

print(f"Predicted minimum on test : {y_pred.min():.2f}")
print(f"Predicted maximum on test : {y_pred.max():.2f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("SAMPLE UNSEEN STUDENTS")
print("=" * 75)

sample = X_test.copy()

sample["Actual_Exam_Score"] = y_test
sample["Predicted_Exam_Score"] = y_pred
sample["Absolute_Error"] = abs(
    sample["Actual_Exam_Score"]
    - sample["Predicted_Exam_Score"]
)

print(
    sample
    .head(10)
    .to_string(index=False)
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("FINAL EVALUATION COMPLETED")
print("=" * 75)