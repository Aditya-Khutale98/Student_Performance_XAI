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
# CANDIDATE DATASET MODEL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE DATASET - MODEL VALIDATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv(
    "data/raw/student_exam_scores.csv"
)

print("\nDataset loaded successfully.")

print(
    "Number of students:",
    len(data)
)


# ============================================================
# 2. SELECT FINAL PROJECT FEATURES
# ============================================================

features = [
    "hours_studied",
    "attendance_percent",
    "previous_scores"
]

target = "exam_score"

X = data[features]

y = data[target]


print("\n" + "=" * 70)
print("FINAL PROJECT FEATURES")
print("=" * 70)

for feature in features:
    print("•", feature)

print("\nTarget:")
print("•", target)


# ============================================================
# 3. BASIC DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print(
    "\nMissing values:"
)

print(
    X.isnull().sum()
)

print(
    "\nTarget missing values:",
    y.isnull().sum()
)

print(
    "\nDuplicate rows:",
    data.duplicated().sum()
)


# ============================================================
# 4. FEATURE CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-TARGET CORRELATION")
print("=" * 70)

for feature in features:

    correlation = X[feature].corr(y)

    print(
        f"{feature:25s}: {correlation:.4f}"
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

print(
    "\n5 folds × 10 repeats = 50 validation runs"
)


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

    rmse_scores = np.sqrt(
        -mse_scores
    )

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

        "MAE": average_mae,

        "MAE Std": std_mae,

        "RMSE": average_rmse,

        "RMSE Std": std_rmse,

        "R2": average_r2,

        "R2 Std": std_r2
    })


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="MAE"
)


print("\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE":
                "{:.3f}".format,

            "MAE Std":
                "{:.3f}".format,

            "RMSE":
                "{:.3f}".format,

            "RMSE Std":
                "{:.3f}".format,

            "R2":
                "{:.4f}".format,

            "R2 Std":
                "{:.4f}".format
        }
    )
)


# ============================================================
# 8. PROPER BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE VALIDATION")
print("=" * 70)

print(
    "\nBaseline = mean score of the training portion "
    "of each validation fold."
)


baseline_mae_scores = []

baseline_rmse_scores = []

baseline_r2_scores = []


for train_index, test_index in cv.split(X):

    X_train = X.iloc[train_index]

    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]

    y_test = y.iloc[test_index]


    # Mean of training target
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


    baseline_mae_scores.append(
        mae
    )

    baseline_rmse_scores.append(
        rmse
    )

    baseline_r2_scores.append(
        r2
    )


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
    "\nLowest average MAE model:"
)

print(
    f"Model: {best_model['Model']}"
)

print(
    f"Model MAE: {best_model['MAE']:.3f}"
)

print(
    f"Baseline MAE: {baseline_mae:.3f}"
)


improvement = (
    (
        baseline_mae
        -
        best_model["MAE"]
    )
    /
    baseline_mae
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
    "\nThe exam scores will be randomly shuffled "
    "10 different times."
)

print(
    "A useful relationship should disappear "
    "after randomization."
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
        "Randomized target does not show "
        "a useful predictive relationship."
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "Randomized target still shows "
        "some apparent predictive signal."
    )


# ============================================================
# 11. IMPORTANT LIMITATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET LIMITATION")
print("=" * 70)

print(
    "\nThis dataset has 200 student records."
)

print(
    "The dataset is useful for developing and "
    "evaluating the project pipeline."
)

print(
    "Its external provenance is not independently "
    "verified from a primary institutional data repository."
)

print(
    "\nTherefore, we will not claim that the model "
    "is universally accurate for all students."
)


# ============================================================
# 12. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nNo existing dataset was replaced."
)

print(
    "No existing model was changed."
)

print(
    "\nNext step will depend on these validation results."
)

print()