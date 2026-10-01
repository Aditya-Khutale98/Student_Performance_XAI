import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/raw/StudentPerformanceFactors.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. SELECT ONLY OUR FINAL FEATURES
# ============================================================

# Dataset stores study time as hours per week.
# Convert it to a daily equivalent.
df["Study_Hours_Daily"] = df["Hours_Studied"] / 7

features = [
    "Study_Hours_Daily",
    "Attendance",
    "Previous_Scores"
]

target = "Exam_Score"


X = df[features].copy()
y = df[target].copy()


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=features
)


# ============================================================
# 4. MODELS
# ============================================================

models = {

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(
            degree=2,
            include_bias=False
        )),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ]),

    "Extra Trees": Pipeline([
        ("model", ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])
}


# ============================================================
# 5. REPEATED CROSS VALIDATION
# ============================================================

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)


# ============================================================
# 6. EVALUATE
# ============================================================

print("=" * 75)
print("STUDENT PERFORMANCE MODEL VALIDATION")
print("=" * 75)

print(f"\nRows used: {len(X)}")
print(f"Features: {features}")
print(f"Target: {target}")

print("\nStudy hours are converted:")
print("Hours_Studied per week / 7 = daily equivalent")


results = []


for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        },
        n_jobs=-1
    )

    mae = -scores["test_MAE"]
    rmse = -scores["test_RMSE"]
    r2 = scores["test_R2"]

    results.append({
        "Model": name,
        "MAE Mean": mae.mean(),
        "MAE Std": mae.std(),
        "RMSE Mean": rmse.mean(),
        "RMSE Std": rmse.std(),
        "R2 Mean": r2.mean(),
        "R2 Std": r2.std()
    })


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE Mean"
)

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 8. BASELINE
# ============================================================

from sklearn.dummy import DummyRegressor

baseline = DummyRegressor(strategy="mean")

baseline_scores = cross_validate(
    baseline,
    X,
    y,
    cv=cv,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    },
    n_jobs=-1
)

baseline_mae = -baseline_scores["test_MAE"].mean()
baseline_rmse = -baseline_scores["test_RMSE"].mean()
baseline_r2 = baseline_scores["test_R2"].mean()

print("\n" + "=" * 75)
print("BASELINE MODEL")
print("=" * 75)

print(f"Baseline MAE  : {baseline_mae:.4f}")
print(f"Baseline RMSE : {baseline_rmse:.4f}")
print(f"Baseline R²   : {baseline_r2:.4f}")


# ============================================================
# 9. FINAL FEATURE CORRELATIONS
# ============================================================

print("\n" + "=" * 75)
print("CORRELATION WITH EXAM SCORE")
print("=" * 75)

correlation = df[
    ["Study_Hours_Daily", "Attendance", "Previous_Scores", "Exam_Score"]
].corr()["Exam_Score"].sort_values(
    ascending=False
)

print(correlation.to_string())


# ============================================================
# 10. INPUT RANGES
# ============================================================

print("\n" + "=" * 75)
print("TRAINING DATA RANGES")
print("=" * 75)

for column in features + [target]:

    print(
        f"{column:25s} "
        f"min={df[column].min():.2f}  "
        f"max={df[column].max():.2f}  "
        f"mean={df[column].mean():.2f}"
    )


print("\n" + "=" * 75)
print("VALIDATION COMPLETED")
print("=" * 75)