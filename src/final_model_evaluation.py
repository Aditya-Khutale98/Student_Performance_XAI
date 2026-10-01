import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/raw/student_exam_scores.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

print(f"\nTotal records: {len(df)}")


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

FEATURES = [
    "hours_studied",
    "attendance_percent",
    "previous_scores"
]

TARGET = "exam_score"

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 3. BASIC DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA CHECK")
print("=" * 70)

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")

print(f"\nTarget:")
print(f"  - {TARGET}")

print("\nMissing values:")
print(X.isnull().sum())

print(f"\nTarget missing values: {y.isnull().sum()}")

print(f"\nDuplicate rows: {df.duplicated().sum()}")


# ============================================================
# 4. CREATE FINAL TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining records : {len(X_train)}")
print(f"Testing records  : {len(X_test)}")

print("\nThe test set will remain untouched until final evaluation.")


# ============================================================
# 5. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linear", LinearRegression())
    ]),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 6. CROSS-VALIDATION ON TRAINING DATA ONLY
# ============================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION ON TRAINING DATA")
print("=" * 70)

print("\n5 folds × 10 repeats = 50 validation runs")

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

cv_results = []

for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    mae = -scores["test_MAE"]
    rmse = -scores["test_RMSE"]
    r2 = scores["test_R2"]

    cv_results.append({
        "Model": name,
        "MAE": mae.mean(),
        "MAE Std": mae.std(),
        "RMSE": rmse.mean(),
        "RMSE Std": rmse.std(),
        "R2": r2.mean(),
        "R2 Std": r2.std()
    })


cv_results_df = pd.DataFrame(cv_results)

print("\n")
print(
    cv_results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "MAE Std": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "RMSE Std": "{:.4f}".format,
            "R2": "{:.4f}".format,
            "R2 Std": "{:.4f}".format
        }
    )
)


# ============================================================
# 7. SELECT MODEL
# ============================================================

# Select based primarily on lowest cross-validation MAE.
best_model_name = cv_results_df.loc[
    cv_results_df["MAE"].idxmin(),
    "Model"
]

print("\n" + "=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print(f"\nSelected model: {best_model_name}")

best_model = models[best_model_name]


# ============================================================
# 8. TRAIN SELECTED MODEL ON ALL TRAINING DATA
# ============================================================

best_model.fit(X_train, y_train)

print("\nModel trained using training data.")


# ============================================================
# 9. FINAL UNSEEN TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL UNSEEN TEST RESULTS")
print("=" * 70)

y_pred = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred)

test_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

test_r2 = r2_score(y_test, y_pred)

print(f"\nTest MAE  : {test_mae:.4f}")
print(f"Test RMSE : {test_rmse:.4f}")
print(f"Test R²   : {test_r2:.4f}")


# ============================================================
# 10. TEST PREDICTIONS
# ============================================================

test_results = X_test.copy()

test_results["Actual Exam Score"] = y_test.values
test_results["Predicted Exam Score"] = y_pred
test_results["Absolute Error"] = np.abs(
    test_results["Actual Exam Score"]
    - test_results["Predicted Exam Score"]
)

print("\n" + "=" * 70)
print("SAMPLE FINAL TEST PREDICTIONS")
print("=" * 70)

print(
    test_results.head(10).to_string(
        index=False,
        formatters={
            "Actual Exam Score": "{:.2f}".format,
            "Predicted Exam Score": "{:.2f}".format,
            "Absolute Error": "{:.2f}".format
        }
    )
)


# ============================================================
# 11. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    f"""
The selected model was evaluated using a completely separate
20% test set that was not used for model fitting.

Cross-validation selected:
    {best_model_name}

Final unseen-test performance:
    MAE  = {test_mae:.4f}
    RMSE = {test_rmse:.4f}
    R²   = {test_r2:.4f}

MAE represents the average absolute prediction error in the
same score units as the exam_score target.

R² indicates how much variation in exam_score is explained
by the model on the test data.

These metrics should NOT be interpreted as universal accuracy
for every student or every educational institution.
"""
)


# ============================================================
# 12. DATASET LIMITATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET LIMITATION")
print("=" * 70)

print(
    """
Dataset size:
    200 student records.

The dataset is useful for developing and evaluating the
student-performance prediction pipeline.

However, its external provenance has not been independently
verified from a primary institutional data repository.

Therefore, this project will not claim universal real-world
accuracy.

Additional institutional or real student data would be needed
to establish stronger external generalization.
"""
)


print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION COMPLETE")
print("=" * 70)