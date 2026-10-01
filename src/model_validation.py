import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("data/raw/student_exam_performance.csv")

print("\n" + "=" * 70)
print("STUDENT PERFORMANCE MODEL - ROBUST VALIDATION")
print("=" * 70)

print("\nDataset loaded successfully.")
print("Number of students:", len(data))


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

features = [
    "hours_studied",
    "attendance_percentage",
    "sleep_hours"
]

target = "exam_score"

X = data[features]
y = data[target]


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nFeatures:")
for feature in features:
    print(" •", feature)

print("\nTarget:")
print(" •", target)

print("\nMissing values:")
print(X.isnull().sum())

print("\nTarget statistics:")
print(y.describe())


# ============================================================
# 4. FEATURE-TARGET CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-TARGET CORRELATION")
print("=" * 70)

for feature in features:
    correlation = X[feature].corr(y)

    print(
        f"{feature:30s}: "
        f"{correlation:.4f}"
    )


# ============================================================
# 5. MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Polynomial Regression":
        make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        )
}


# ============================================================
# 6. REPEATED CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("REPEATED CROSS-VALIDATION")
print("=" * 70)

print("\nUsing:")
print("5 folds × 10 repeats = 50 validation runs")

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


results = []


for name, model in models.items():

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error"
    )

    mae_scores = -mae_scores

    average_mae = mae_scores.mean()
    std_mae = mae_scores.std()


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    mse_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error"
    )

    rmse_scores = np.sqrt(-mse_scores)

    average_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()


    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    r2_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2"
    )

    average_r2 = r2_scores.mean()
    std_r2 = r2_scores.std()


    results.append({

        "Model": name,

        "MAE":
            average_mae,

        "MAE Std":
            std_mae,

        "RMSE":
            average_rmse,

        "RMSE Std":
            std_rmse,

        "R2":
            average_r2,

        "R2 Std":
            std_r2
    })


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE"
)


print("\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.3f}".format,
            "MAE Std": "{:.3f}".format,
            "RMSE": "{:.3f}".format,
            "RMSE Std": "{:.3f}".format,
            "R2": "{:.4f}".format,
            "R2 Std": "{:.4f}".format
        }
    )
)


# ============================================================
# 8. PROPER BASELINE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BASELINE VALIDATION")
print("=" * 70)

print(
    "\nBaseline = predict the average score "
    "of the training portion of each fold."
)

baseline_mae_scores = []

baseline_rmse_scores = []

baseline_r2_scores = []


for train_index, test_index in cv.split(X):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]


    # Mean score from TRAINING data only
    baseline_prediction = y_train.mean()


    predictions = np.full(
        len(y_test),
        baseline_prediction
    )


    # MAE
    mae = np.mean(
        np.abs(
            y_test - predictions
        )
    )


    # RMSE
    rmse = np.sqrt(
        np.mean(
            (y_test - predictions) ** 2
        )
    )


    # R²
    ss_res = np.sum(
        (y_test - predictions) ** 2
    )

    ss_tot = np.sum(
        (y_test - y_test.mean()) ** 2
    )


    if ss_tot != 0:

        r2 = 1 - (
            ss_res / ss_tot
        )

    else:

        r2 = 0


    baseline_mae_scores.append(mae)

    baseline_rmse_scores.append(rmse)

    baseline_r2_scores.append(r2)


baseline_mae = np.mean(
    baseline_mae_scores
)

baseline_rmse = np.mean(
    baseline_rmse_scores
)

baseline_r2 = np.mean(
    baseline_r2_scores
)


print(
    f"\nBaseline MAE  : {baseline_mae:.3f}"
)

print(
    f"Baseline RMSE : {baseline_rmse:.3f}"
)

print(
    f"Baseline R²   : {baseline_r2:.4f}"
)


# ============================================================
# 9. MODEL VS BASELINE
# ============================================================

print("\n" + "=" * 70)
print("MODEL VS BASELINE")
print("=" * 70)

best_model = results_df.iloc[0]

print(
    f"\nLowest average MAE model:"
)

print(
    f"Model : {best_model['Model']}"
)

print(
    f"Model MAE : {best_model['MAE']:.3f}"
)

print(
    f"Baseline MAE : {baseline_mae:.3f}"
)

improvement = (
    (baseline_mae - best_model["MAE"])
    / baseline_mae
) * 100


print(
    f"\nMAE improvement over baseline: "
    f"{improvement:.2f}%"
)


# ============================================================
# 10. TARGET SHUFFLING TEST
# ============================================================

print("\n" + "=" * 70)
print("TARGET-SHUFFLING TEST")
print("=" * 70)

print(
    "\nThe exam scores will be randomly shuffled."
)

print(
    "If the relationship disappears, "
    "the original relationship is more credible."
)


shuffle_r2_values = []


for seed in range(10):

    shuffled_y = (
        y.sample(
            frac=1,
            random_state=seed
        )
        .reset_index(drop=True)
    )

    shuffled_scores = cross_val_score(

        LinearRegression(),

        X,

        shuffled_y,

        cv=cv,

        scoring="r2"
    )

    shuffle_r2_values.append(
        shuffled_scores.mean()
    )


shuffle_mean = np.mean(
    shuffle_r2_values
)

shuffle_std = np.std(
    shuffle_r2_values
)


print(
    f"\nAverage shuffled R²: "
    f"{shuffle_mean:.4f}"
)

print(
    f"Shuffled R² Std: "
    f"{shuffle_std:.4f}"
)


if shuffle_mean <= 0:

    print(
        "\nGOOD SIGN:"
    )

    print(
        "Randomized target does not "
        "show a useful predictive relationship."
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "Randomized target is showing "
        "some predictive signal."
    )


# ============================================================
# 11. INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION INTERPRETATION")
print("=" * 70)

print(
    "\nImportant:"
)

print(
    "• MAE shows the average prediction error in marks."
)

print(
    "• RMSE gives more penalty to large prediction errors."
)

print(
    "• R² describes how much variation the model explains."
)

print(
    "• Cross-validation checks whether performance "
    "is stable across different data splits."
)

print(
    "• Target shuffling checks whether the relationship "
    "disappears when the target is randomized."
)

print(
    "\nDataset limitation:"
)

print(
    "The current dataset is synthetic."
)

print(
    "Therefore, these results should NOT be presented "
    "as proof of real-world prediction accuracy."
)


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nNext stage after validation:"
)

print(
    "1. Select a defensible final model"
)

print(
    "2. Measure prediction uncertainty"
)

print(
    "3. Connect model + SHAP"
)

print(
    "4. Build Streamlit interface"
)

print()