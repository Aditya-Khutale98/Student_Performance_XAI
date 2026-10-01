import os
import sys
import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ============================================================
# MAKE SRC DIRECTORY AVAILABLE
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(
        0,
        CURRENT_DIR
    )


# ============================================================
# IMPORT FINAL MODEL
# ============================================================

from bounded_model import BoundedRegression


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/raw/StudentPerformanceFactors.csv"
)

MODEL_PATH = (
    "models/student_performance_bounded.pkl"
)


# ============================================================
# FINAL FEATURES
# ============================================================

FEATURES = [
    "Study_Hours_Daily",
    "Attendance",
    "Previous_Scores"
]


TARGET = "Exam_Score"


# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_NAMES = {

    "Study_Hours_Daily":
        "Study Hours",

    "Attendance":
        "Attendance",

    "Previous_Scores":
        "Previous Performance"
}


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# CLEAN TARGET
# ============================================================

# Exam score should be on a 0-100 scale.
# The dataset contains one value of 101.
df[TARGET] = df[TARGET].clip(
    0,
    100
)


# ============================================================
# CONVERT WEEKLY STUDY HOURS → DAILY STUDY HOURS
# ============================================================

# Hours_Studied in the dataset represents
# weekly study hours.

df["Study_Hours_Daily"] = (
    df["Hours_Studied"] / 7.0
)


# ============================================================
# LOAD FINAL BOUNDED MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# SHAP BACKGROUND DATA
# ============================================================

background_data = (
    df[FEATURES]
    .sample(
        n=min(100, len(df)),
        random_state=42
    )
    .reset_index(drop=True)
)


# ============================================================
# MODEL PREDICTION FUNCTION FOR SHAP
# ============================================================

def model_predict(data):

    if isinstance(
        data,
        np.ndarray
    ):

        data = pd.DataFrame(
            data,
            columns=FEATURES
        )

    elif not isinstance(
        data,
        pd.DataFrame
    ):

        data = pd.DataFrame(
            data,
            columns=FEATURES
        )

    return np.asarray(
        model.predict(
            data[FEATURES]
        ),
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
# TRAINING DATA STATISTICS
# ============================================================

TRAINING_STATISTICS = {

    "Study_Hours_Daily": {

        "min":
            float(
                df[
                    "Study_Hours_Daily"
                ].min()
            ),

        "max":
            float(
                df[
                    "Study_Hours_Daily"
                ].max()
            ),

        "mean":
            float(
                df[
                    "Study_Hours_Daily"
                ].mean()
            ),

        "median":
            float(
                df[
                    "Study_Hours_Daily"
                ].median()
            )
    },


    "Attendance": {

        "min":
            float(
                df[
                    "Attendance"
                ].min()
            ),

        "max":
            float(
                df[
                    "Attendance"
                ].max()
            ),

        "mean":
            float(
                df[
                    "Attendance"
                ].mean()
            ),

        "median":
            float(
                df[
                    "Attendance"
                ].median()
            )
    },


    "Previous_Scores": {

        "min":
            float(
                df[
                    "Previous_Scores"
                ].min()
            ),

        "max":
            float(
                df[
                    "Previous_Scores"
                ].max()
            ),

        "mean":
            float(
                df[
                    "Previous_Scores"
                ].mean()
            ),

        "median":
            float(
                df[
                    "Previous_Scores"
                ].median()
            )
    }
}


# ============================================================
# STUDY TIME CONVERSION
# ============================================================

def convert_study_time(time_value):
    """
    Convert HH.MM study time into decimal hours.

    Examples:

        3.45  -> 3 hours 45 minutes -> 3.75
        2.30  -> 2 hours 30 minutes -> 2.50
        5     -> 5 hours            -> 5.00
        24.00 -> 24 hours            -> 24.00

    IMPORTANT:
    3.45 means 3 hours 45 minutes,
    NOT 3.45 decimal hours.
    """

    time_text = str(
        time_value
    ).strip()


    if not time_text:

        raise ValueError(
            "Study time cannot be empty."
        )


    # --------------------------------------------------------
    # HH.MM FORMAT
    # --------------------------------------------------------

    if "." in time_text:

        parts = time_text.split(".")


        if len(parts) != 2:

            raise ValueError(
                "Enter study time in HH.MM format, "
                "for example 3.45."
            )


        hours_text = parts[0]

        minutes_text = parts[1]


        if not hours_text.isdigit():

            raise ValueError(
                "Hours must be a non-negative whole number."
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


    # --------------------------------------------------------
    # WHOLE HOURS
    # --------------------------------------------------------

    else:

        if not time_text.isdigit():

            raise ValueError(
                "Enter study time in HH.MM format, "
                "for example 3.45."
            )


        hours = int(
            time_text
        )

        minutes = 0


    # --------------------------------------------------------
    # VALIDATE HOURS
    # --------------------------------------------------------

    if hours < 0 or hours > 24:

        raise ValueError(
            "Hours must be between 0 and 24."
        )


    # --------------------------------------------------------
    # VALIDATE MINUTES
    # --------------------------------------------------------

    if minutes < 0 or minutes >= 60:

        raise ValueError(
            "Minutes must be between 00 and 59."
        )


    # --------------------------------------------------------
    # 24:01, 24:30 etc. are invalid
    # --------------------------------------------------------

    if hours == 24 and minutes > 0:

        raise ValueError(
            "Study time cannot be greater than 24 hours."
        )


    return (
        hours + minutes / 60.0
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs(
    study_time,
    attendance_percent,
    previous_scores
):
    """
    Validate user inputs.

    Accepted ranges:

        Study Hours  : 0-24 hours/day
        Attendance   : 0-100%
        Previous     : 0-100
    """

    # --------------------------------------------------------
    # Study time
    # --------------------------------------------------------

    decimal_hours = convert_study_time(
        study_time
    )


    # --------------------------------------------------------
    # Attendance
    # --------------------------------------------------------

    try:

        attendance = float(
            attendance_percent
        )

    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            "Attendance must be a number."
        )


    if attendance < 0 or attendance > 100:

        raise ValueError(
            "Attendance must be between 0 and 100."
        )


    # --------------------------------------------------------
    # Previous performance
    # --------------------------------------------------------

    try:

        previous_score = float(
            previous_scores
        )

    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            "Previous Performance must be a number."
        )


    if previous_score < 0 or previous_score > 100:

        raise ValueError(
            "Previous Performance must be between 0 and 100."
        )


    return (
        decimal_hours,
        attendance,
        previous_score
    )


# ============================================================
# GENERATE PREDICTION
# ============================================================

def generate_prediction(
    decimal_hours,
    attendance,
    previous_score
):

    student_data = pd.DataFrame(

        [[
            decimal_hours,
            attendance,
            previous_score
        ]],

        columns=FEATURES
    )


    prediction = model_predict(
        student_data
    )[0]


    # Final safety bound

    prediction = float(
        np.clip(
            prediction,
            0,
            100
        )
    )


    return (
        prediction,
        student_data
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def generate_shap_explanation(
    student_data
):

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


    explanations = []


    for feature, value in zip(
        FEATURES,
        shap_values
    ):

        explanations.append({

            "feature":
                feature,

            "feature_name":
                FEATURE_NAMES[feature],

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


    return (
        base_value,
        explanations
    )


# ============================================================
# DATA COVERAGE INFORMATION
# ============================================================

def generate_data_coverage(
    decimal_hours,
    attendance,
    previous_score
):
    """
    Checks whether inputs fall inside the ranges observed
    in the training data.

    IMPORTANT:
    This does NOT block the prediction.

    The user can still enter any valid physical value.
    """

    values = {

        "Study_Hours_Daily":
            decimal_hours,

        "Attendance":
            attendance,

        "Previous_Scores":
            previous_score
    }


    coverage = []


    for feature, value in values.items():

        minimum = (
            TRAINING_STATISTICS[
                feature
            ]["min"]
        )

        maximum = (
            TRAINING_STATISTICS[
                feature
            ]["max"]
        )


        if value < minimum or value > maximum:

            coverage.append({

                "feature":
                    FEATURE_NAMES[feature],

                "value":
                    value,

                "training_min":
                    minimum,

                "training_max":
                    maximum,

                "message":
                    (
                        f"{FEATURE_NAMES[feature]} "
                        f"({value:.2f}) is outside "
                        f"the range observed in the "
                        f"training data "
                        f"({minimum:.2f} to "
                        f"{maximum:.2f}). "
                        f"The model will still provide "
                        f"a prediction, but this input "
                        f"represents extrapolation beyond "
                        f"the observed training range."
                    )
            })


    return coverage


# ============================================================
# SHAP-BASED FACTORS
# ============================================================

def generate_factors(
    shap_explanations
):

    positive = []

    negative = []


    for item in shap_explanations:

        value = item[
            "shap_value"
        ]

        name = item[
            "feature_name"
        ]


        if value > 0:

            positive.append({

                "feature":
                    name,

                "shap_value":
                    value,

                "message":
                    (
                        f"{name} contributed "
                        f"+{value:.2f} score points "
                        f"to the prediction."
                    )
            })


        elif value < 0:

            negative.append({

                "feature":
                    name,

                "shap_value":
                    value,

                "message":
                    (
                        f"{name} contributed "
                        f"{value:.2f} score points "
                        f"to the prediction."
                    )
            })


    return (
        positive,
        negative
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    decimal_hours,
    attendance,
    previous_score,
    negative_factors
):

    recommendations = []


    study_median = (
        TRAINING_STATISTICS[
            "Study_Hours_Daily"
        ]["median"]
    )


    previous_median = (
        TRAINING_STATISTICS[
            "Previous_Scores"
        ]["median"]
    )


    # --------------------------------------------------------
    # Study hours
    # --------------------------------------------------------

    if decimal_hours < study_median:

        recommendations.append(
            "Consider gradually increasing "
            "daily study time and maintaining "
            "a consistent study schedule."
        )


    # --------------------------------------------------------
    # Attendance
    # --------------------------------------------------------

    if attendance < 75:

        recommendations.append(
            "Attendance is relatively low. "
            "Try to maintain more regular "
            "class attendance."
        )

    elif attendance < 85:

        recommendations.append(
            "Attendance is moderate. "
            "Maintaining regular attendance "
            "could support better performance."
        )


    # --------------------------------------------------------
    # Previous performance
    # --------------------------------------------------------

    if previous_score < previous_median:

        recommendations.append(
            "Previous performance is below "
            "the dataset median. Focus on "
            "weak subjects, revision, and "
            "practice of previous topics."
        )


    # --------------------------------------------------------
    # SHAP-based recommendations
    # --------------------------------------------------------

    for factor in negative_factors:

        feature = factor[
            "feature"
        ]


        if feature == "Study Hours":

            recommendations.append(
                "Study Hours are negatively "
                "contributing to this prediction. "
                "A consistent study routine may "
                "help improve preparation."
            )


        elif feature == "Attendance":

            recommendations.append(
                "Attendance is negatively "
                "contributing to this prediction. "
                "Try to maintain more regular "
                "attendance."
            )


        elif feature == "Previous Performance":

            recommendations.append(
                "Previous Performance is negatively "
                "contributing to the prediction. "
                "Additional revision and practice "
                "may help strengthen preparation."
            )


    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = []


    for recommendation in recommendations:

        if recommendation not in unique:

            unique.append(
                recommendation
            )


    return unique


# ============================================================
# PREDICTION INDICATOR
# ============================================================

def get_prediction_indicator(
    prediction
):
    """
    Simple visual indicator for the predicted score.

    This is only a user-interface indicator.
    It is not an accuracy/confidence score.
    """

    if prediction < 60:

        return {

            "level":
                "LOWER",

            "symbol":
                "🔴",

            "message":
                "The predicted score is in a "
                "lower range on the 0-100 score scale."
        }


    elif prediction < 75:

        return {

            "level":
                "MODERATE",

            "symbol":
                "🟡",

            "message":
                "The predicted score is in a "
                "moderate range on the 0-100 score scale."
        }


    else:

        return {

            "level":
                "HIGHER",

            "symbol":
                "🟢",

            "message":
                "The predicted score is in a "
                "higher range on the 0-100 score scale."
        }


# ============================================================
# EMPIRICAL UNCERTAINTY
# ============================================================

def calculate_uncertainty():

    """
    Estimate prediction error using the same bounded
    regression approach on an unseen 20% test set.

    This is an empirical error estimate, NOT a guaranteed
    confidence interval.
    """

    X = df[
        FEATURES
    ].copy()


    y = df[
        TARGET
    ].copy()


    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42
        )
    )


    # --------------------------------------------------------
    # Train a fresh bounded validation model
    # --------------------------------------------------------

    validation_model = (
        BoundedRegression()
    )


    validation_model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    predictions = (
        validation_model.predict(
            X_test
        )
    )


    # --------------------------------------------------------
    # Absolute errors
    # --------------------------------------------------------

    absolute_errors = np.abs(
        y_test.values -
        predictions
    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = float(
        np.mean(
            absolute_errors
        )
    )


    # --------------------------------------------------------
    # 90th percentile absolute error
    # --------------------------------------------------------

    error_90 = float(
        np.percentile(
            absolute_errors,
            90
        )
    )


    return {

        "mae":
            mae,

        "error_90":
            error_90
    }


# ============================================================
# CALCULATE UNCERTAINTY ONCE
# ============================================================

UNCERTAINTY = (
    calculate_uncertainty()
)


# ============================================================
# MASTER FUNCTION
# ============================================================

def predict_student_performance(
    study_time,
    attendance_percent,
    previous_scores
):

    # --------------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------------

    (
        decimal_hours,
        attendance,
        previous_score

    ) = validate_inputs(

        study_time,
        attendance_percent,
        previous_scores
    )


    # --------------------------------------------------------
    # 2. Generate prediction
    # --------------------------------------------------------

    (
        prediction,
        student_data

    ) = generate_prediction(

        decimal_hours,
        attendance,
        previous_score
    )


    # --------------------------------------------------------
    # 3. Generate SHAP explanation
    # --------------------------------------------------------

    (
        base_value,
        shap_explanations

    ) = generate_shap_explanation(
        student_data
    )


    # --------------------------------------------------------
    # 4. Positive / negative factors
    # --------------------------------------------------------

    (
        positive_factors,
        negative_factors

    ) = generate_factors(
        shap_explanations
    )


    # --------------------------------------------------------
    # 5. Data coverage information
    # --------------------------------------------------------

    data_coverage = (
        generate_data_coverage(

            decimal_hours,
            attendance,
            previous_score
        )
    )


    # --------------------------------------------------------
    # 6. Recommendations
    # --------------------------------------------------------

    recommendations = (
        generate_recommendations(

            decimal_hours,
            attendance,
            previous_score,

            negative_factors
        )
    )


    # --------------------------------------------------------
    # 7. Prediction indicator
    # --------------------------------------------------------

    indicator = (
        get_prediction_indicator(
            prediction
        )
    )


    # --------------------------------------------------------
    # 8. Empirical uncertainty
    # --------------------------------------------------------

    error_90 = (
        UNCERTAINTY[
            "error_90"
        ]
    )


    lower_bound = max(
        0.0,
        prediction - error_90
    )


    upper_bound = min(
        100.0,
        prediction + error_90
    )


    # --------------------------------------------------------
    # 9. Final result
    # --------------------------------------------------------

    return {

        "input": {

            "study_time":
                str(study_time),

            "study_hours":
                float(decimal_hours),

            "attendance":
                float(attendance),

            "previous_score":
                float(previous_score)
        },


        "prediction": {

            "score":
                float(prediction),

            "level":
                indicator[
                    "level"
                ],

            "symbol":
                indicator[
                    "symbol"
                ],

            "message":
                indicator[
                    "message"
                ]
        },


        "shap": {

            "base_value":
                float(base_value),

            "contributions":
                shap_explanations,

            "positive_factors":
                positive_factors,

            "negative_factors":
                negative_factors
        },


        "data_coverage":
            data_coverage,


        "recommendations":
            recommendations,


        "uncertainty": {

            "typical_mae":
                float(
                    UNCERTAINTY[
                        "mae"
                    ]
                ),

            "error_90_percentile":
                float(error_90),

            "lower_bound":
                float(lower_bound),

            "upper_bound":
                float(upper_bound),

            "message":
                (
                    "This range is an empirical "
                    "estimate based on validation "
                    "errors. It is not a guaranteed "
                    "prediction interval, and the actual "
                    "student score may be different."
                )
        }
    }


# ============================================================
# TEST THE COMPLETE SYSTEM
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 75
    )

    print(
        "STUDENT PERFORMANCE "
        "PREDICTION SYSTEM"
    )

    print(
        "=" * 75
    )


    # --------------------------------------------------------
    # Test student
    # --------------------------------------------------------

    study_time = "3.45"

    attendance = 85

    previous_score = 70


    try:

        result = (
            predict_student_performance(

                study_time,
                attendance,
                previous_score
            )
        )


        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        print("\nINPUT")

        print(
            "-" * 75
        )


        print(
            f"Study Time: "
            f"{result['input']['study_time']}"
        )


        print(
            f"Converted Study Hours: "
            f"{result['input']['study_hours']:.2f}"
        )


        print(
            f"Attendance: "
            f"{result['input']['attendance']:.2f}%"
        )


        print(
            f"Previous Performance: "
            f"{result['input']['previous_score']:.2f}"
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        print("\nPREDICTION")

        print(
            "-" * 75
        )


        print(
            f"Predicted Exam Score: "
            f"{result['prediction']['score']:.2f}"
        )


        print(
            f"{result['prediction']['symbol']} "
            f"{result['prediction']['level']}"
        )


        print(
            result[
                "prediction"
            ]["message"]
        )


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        print("\nSHAP CONTRIBUTIONS")

        print(
            "-" * 75
        )


        print(
            f"Base Value: "
            f"{result['shap']['base_value']:.2f}"
        )


        for item in result[
            "shap"
        ]["contributions"]:

            print(
                f"{item['feature_name']:25s} "
                f"{item['shap_value']:+.4f}"
            )


        # ----------------------------------------------------
        # POSITIVE FACTORS
        # ----------------------------------------------------

        print("\nPOSITIVE FACTORS")

        print(
            "-" * 75
        )


        if result[
            "shap"
        ]["positive_factors"]:

            for factor in result[
                "shap"
            ]["positive_factors"]:

                print(
                    "🟢 " +
                    factor["message"]
                )

        else:

            print("None")


        # ----------------------------------------------------
        # NEGATIVE FACTORS
        # ----------------------------------------------------

        print("\nNEGATIVE FACTORS")

        print(
            "-" * 75
        )


        if result[
            "shap"
        ]["negative_factors"]:

            for factor in result[
                "shap"
            ]["negative_factors"]:

                print(
                    "🔴 " +
                    factor["message"]
                )

        else:

            print("None")


        # ----------------------------------------------------
        # DATA COVERAGE
        # ----------------------------------------------------

        print("\nDATA COVERAGE")

        print(
            "-" * 75
        )


        if result[
            "data_coverage"
        ]:

            for item in result[
                "data_coverage"
            ]:

                print(
                    "⚠️ " +
                    item["message"]
                )

        else:

            print(
                "All inputs are within the "
                "ranges observed in the training data."
            )


        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        print("\nRECOMMENDATIONS")

        print(
            "-" * 75
        )


        if result[
            "recommendations"
        ]:

            for recommendation in result[
                "recommendations"
            ]:

                print(
                    "• " +
                    recommendation
                )

        else:

            print("None")


        # ----------------------------------------------------
        # UNCERTAINTY
        # ----------------------------------------------------

        print("\nPREDICTION UNCERTAINTY")

        print(
            "-" * 75
        )


        print(
            f"Typical validation MAE: "
            f"±{result['uncertainty']['typical_mae']:.2f}"
        )


        print(
            f"Empirical 90th-percentile "
            f"error range: "
            f"{result['uncertainty']['lower_bound']:.2f} "
            f"to "
            f"{result['uncertainty']['upper_bound']:.2f}"
        )


        print(
            "⚠️ " +
            result[
                "uncertainty"
            ]["message"]
        )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print(
            "\n" + "=" * 75
        )


        print(
            "COMPLETE SYSTEM TEST SUCCESSFUL"
        )


        print(
            "=" * 75
        )


    except Exception as error:

        print("\nERROR")

        print(
            "-" * 75
        )


        print(
            type(error).__name__,
            ":",
            error
        )


        print(
            "\nPlease check the input values."
        )