import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap


# ============================================================
# MAKE SRC IMPORTABLE
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(
        0,
        CURRENT_DIR
    )


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/student_performance_bounded.pkl"
)

DATA_PATH = (
    "data/raw/StudentPerformanceFactors.csv"
)

FEATURES = [
    "Study_Hours_Daily",
    "Attendance",
    "Previous_Scores"
]


# ============================================================
# LOAD FINAL MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# CLEAN DATASET
# ============================================================

# Exam score is bounded between 0 and 100
df["Exam_Score"] = df["Exam_Score"].clip(
    0,
    100
)


# Original Hours_Studied is weekly
# Convert to daily hours
df["Study_Hours_Daily"] = (
    df["Hours_Studied"] / 7.0
)


# ============================================================
# SHAP BACKGROUND DATA
# ============================================================

# We do not need all 6607 students as the SHAP background.
# A representative sample makes SHAP much faster.

background_data = (
    df[FEATURES]
    .sample(
        n=min(100, len(df)),
        random_state=42
    )
    .reset_index(drop=True)
)


# ============================================================
# MODEL PREDICTION FUNCTION
# ============================================================

def model_predict(data):

    # SHAP may provide a NumPy array.
    # Convert it back to DataFrame with feature names.

    if isinstance(data, np.ndarray):

        data = pd.DataFrame(
            data,
            columns=FEATURES
        )

    elif not isinstance(data, pd.DataFrame):

        data = pd.DataFrame(
            data,
            columns=FEATURES
        )

    predictions = model.predict(
        data[FEATURES]
    )

    return np.asarray(
        predictions,
        dtype=float
    )


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.Explainer(
    model_predict,
    background_data,
    algorithm="permutation"
)


# ============================================================
# STUDY TIME CONVERSION
# ============================================================

def convert_study_time(time_value):
    """
    Convert HH.MM study time into decimal hours.

    Examples:

        3.45 → 3 hours 45 minutes → 3.75
        2.30 → 2 hours 30 minutes → 2.50
        5    → 5 hours → 5.00
    """

    time_text = str(
        time_value
    ).strip()

    if not time_text:

        raise ValueError(
            "Study time cannot be empty."
        )

    if "." in time_text:

        parts = time_text.split(".")

        if len(parts) != 2:

            raise ValueError(
                "Enter study time in HH.MM format."
            )

        hours_text = parts[0]
        minutes_text = parts[1]

        if not hours_text.isdigit():

            raise ValueError(
                "Hours must be a whole number."
            )

        if not minutes_text.isdigit():

            raise ValueError(
                "Minutes must contain numbers only."
            )

        hours = int(
            hours_text
        )

        minutes = int(
            minutes_text
        )

    else:

        if not time_text.isdigit():

            raise ValueError(
                "Enter study time in HH.MM format."
            )

        hours = int(
            time_text
        )

        minutes = 0

    if minutes < 0 or minutes >= 60:

        raise ValueError(
            "Minutes must be between 00 and 59."
        )

    if hours < 0 or hours > 24:

        raise ValueError(
            "Hours must be between 0 and 24."
        )

    if hours == 24 and minutes > 0:

        raise ValueError(
            "Study time cannot be greater than 24 hours."
        )

    return hours + (
        minutes / 60.0
    )


# ============================================================
# GENERATE SHAP EXPLANATION
# ============================================================

def explain_prediction(
    study_time,
    attendance_percent,
    previous_scores
):

    # --------------------------------------------------------
    # Convert study time
    # --------------------------------------------------------

    decimal_hours = convert_study_time(
        study_time
    )


    # --------------------------------------------------------
    # Validate attendance
    # --------------------------------------------------------

    attendance = float(
        attendance_percent
    )

    if attendance < 0 or attendance > 100:

        raise ValueError(
            "Attendance must be between 0 and 100."
        )


    # --------------------------------------------------------
    # Validate previous performance
    # --------------------------------------------------------

    previous_score = float(
        previous_scores
    )

    if previous_score < 0 or previous_score > 100:

        raise ValueError(
            "Previous performance must be between 0 and 100."
        )


    # --------------------------------------------------------
    # Create student input
    # --------------------------------------------------------

    student_data = pd.DataFrame(
        [[
            decimal_hours,
            attendance,
            previous_score
        ]],
        columns=FEATURES
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model_predict(
        student_data
    )[0]


    # --------------------------------------------------------
    # SHAP calculation
    # --------------------------------------------------------

    shap_result = explainer(
        student_data
    )

    shap_values = (
        shap_result.values[0]
    )

    base_value = float(
        np.asarray(
            shap_result.base_values
        ).reshape(-1)[0]
    )


    # --------------------------------------------------------
    # Store explanations
    # --------------------------------------------------------

    explanations = []

    for feature, value in zip(
        FEATURES,
        shap_values
    ):

        explanations.append({

            "feature": feature,

            "shap_value":
                float(value),

            "absolute_shap":
                abs(float(value))
        })


    # Strongest contribution first

    explanations.sort(
        key=lambda item:
        item["absolute_shap"],
        reverse=True
    )


    return {

        "prediction":
            float(prediction),

        "base_value":
            base_value,

        "study_hours":
            decimal_hours,

        "attendance":
            attendance,

        "previous_score":
            previous_score,

        "shap_explanations":
            explanations
    }


# ============================================================
# HUMAN-READABLE EXPLANATION
# ============================================================

def create_explanation_text(
    explanations
):

    positive = []
    negative = []

    feature_names = {

        "Study_Hours_Daily":
            "Study Hours",

        "Attendance":
            "Attendance",

        "Previous_Scores":
            "Previous Performance"
    }


    for item in explanations:

        feature = item["feature"]

        value = item["shap_value"]

        name = feature_names.get(
            feature,
            feature
        )


        if value > 0:

            positive.append(
                f"{name} contributed "
                f"+{value:.2f} score points "
                f"to the prediction."
            )


        elif value < 0:

            negative.append(
                f"{name} contributed "
                f"{value:.2f} score points "
                f"to the prediction."
            )


        else:

            positive.append(
                f"{name} had almost no "
                f"effect on this prediction."
            )


    return positive, negative


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 75)

    print(
        "STUDENT PERFORMANCE - "
        "BOUNDED MODEL + SHAP"
    )

    print("=" * 75)


    # Test student

    study_time = "3.45"

    attendance = 85

    previous_score = 70


    try:

        result = explain_prediction(
            study_time,
            attendance,
            previous_score
        )


        print("\nINPUT")

        print("-" * 75)

        print(
            f"Study Time       : "
            f"{study_time}"
        )

        print(
            f"Attendance       : "
            f"{attendance}%"
        )

        print(
            f"Previous Score   : "
            f"{previous_score}"
        )


        print("\nPROCESSED INPUT")

        print("-" * 75)

        print(
            f"Study Hours      : "
            f"{result['study_hours']:.2f}"
        )

        print(
            f"Attendance       : "
            f"{result['attendance']:.2f}%"
        )

        print(
            f"Previous Score   : "
            f"{result['previous_score']:.2f}"
        )


        print("\nPREDICTION")

        print("-" * 75)

        print(
            f"Predicted Exam Score : "
            f"{result['prediction']:.2f}"
        )

        print(
            f"SHAP Base Value      : "
            f"{result['base_value']:.2f}"
        )


        print("\nSHAP CONTRIBUTIONS")

        print("-" * 75)

        for item in result[
            "shap_explanations"
        ]:

            print(
                f"{item['feature']:22s} "
                f"{item['shap_value']:+.4f}"
            )


        positive, negative = (
            create_explanation_text(
                result[
                    "shap_explanations"
                ]
            )
        )


        print("\nPOSITIVE CONTRIBUTIONS")

        print("-" * 75)

        if positive:

            for explanation in positive:

                print(
                    "• " + explanation
                )

        else:

            print("None")


        print("\nNEGATIVE CONTRIBUTIONS")

        print("-" * 75)

        if negative:

            for explanation in negative:

                print(
                    "• " + explanation
                )

        else:

            print("None")


        print(
            "\n" + "=" * 75
        )

        print(
            "SHAP TEST COMPLETE"
        )

        print(
            "=" * 75
        )


    except Exception as error:

        print("\nERROR")

        print("-" * 75)

        print(
            type(error).__name__,
            ":",
            error
        )