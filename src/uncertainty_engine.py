import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/student_exam_scores.csv"

MODEL_PATH = "models/student_performance_linear_regression.pkl"

FEATURES = [
    "hours_studied",
    "attendance_percent",
    "previous_scores"
]

TARGET = "exam_score"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# VALIDATION RESIDUALS
# ============================================================

def calculate_validation_errors():

    X = df[FEATURES]
    y = df[TARGET]

    # Same final 80/20 split used during evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Train temporary evaluation model
    validation_model = LinearRegression()

    validation_model.fit(
        X_train,
        y_train
    )

    # Predictions on unseen test data
    predictions = validation_model.predict(
        X_test
    )

    # Residuals
    residuals = y_test.values - predictions

    absolute_errors = np.abs(
        residuals
    )

    return {
        "residuals": residuals,

        "absolute_errors":
            absolute_errors,

        "mae":
            float(
                np.mean(absolute_errors)
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        residuals ** 2
                    )
                )
            )
    }


# ============================================================
# BUILD UNCERTAINTY RANGE
# ============================================================

def calculate_prediction_range(
    prediction
):

    validation = calculate_validation_errors()

    absolute_errors = validation[
        "absolute_errors"
    ]

    # --------------------------------------------------------
    # Typical error
    # --------------------------------------------------------

    mae = validation["mae"]

    # --------------------------------------------------------
    # Conservative error range
    #
    # 90th percentile means approximately 90%
    # of absolute errors in this particular
    # held-out test set were below this value.
    # --------------------------------------------------------

    error_90 = float(
        np.percentile(
            absolute_errors,
            90
        )
    )

    lower = prediction - error_90

    upper = prediction + error_90

    return {

        "prediction": float(
            prediction
        ),

        "typical_error_mae": float(
            mae
        ),

        "error_90_percentile": error_90,

        "lower_bound": float(
            lower
        ),

        "upper_bound": float(
            upper
        )
    }


# ============================================================
# HUMAN-READABLE UNCERTAINTY
# ============================================================

def create_uncertainty_message(
    uncertainty
):

    prediction = uncertainty[
        "prediction"
    ]

    mae = uncertainty[
        "typical_error_mae"
    ]

    lower = uncertainty[
        "lower_bound"
    ]

    upper = uncertainty[
        "upper_bound"
    ]

    return {

        "summary":
            f"Predicted score: "
            f"{prediction:.2f}",

        "typical_error":
            f"Typical validation error: "
            f"approximately ±{mae:.2f} "
            f"score units.",

        "range":
            f"Observed-data uncertainty range: "
            f"{lower:.2f} to {upper:.2f}.",

        "warning":
            "This is an estimate based on the "
            "model's validation errors. The actual "
            "student score may be different."
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "STUDENT PERFORMANCE - "
        "PREDICTION UNCERTAINTY"
    )
    print("=" * 70)


    # Same student used throughout the pipeline

    study_hours = 3.75
    attendance = 85
    previous_score = 70


    # Create input

    student_data = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_score
        ]],
        columns=FEATURES
    )


    # Generate prediction

    prediction = model.predict(
        student_data
    )[0]


    # Calculate uncertainty

    uncertainty = calculate_prediction_range(
        prediction
    )


    # Human-readable output

    message = create_uncertainty_message(
        uncertainty
    )


    print("\nPREDICTION")
    print("-" * 70)

    print(
        message["summary"]
    )


    print("\nMODEL ERROR")
    print("-" * 70)

    print(
        message["typical_error"]
    )


    print("\nOBSERVED-DATA RANGE")
    print("-" * 70)

    print(
        message["range"]
    )


    print("\nIMPORTANT")
    print("-" * 70)

    print(
        "⚠️ " +
        message["warning"]
    )


    print("\n" + "=" * 70)
    print(
        "UNCERTAINTY TEST COMPLETE"
    )
    print("=" * 70)