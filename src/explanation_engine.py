import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/student_exam_scores.csv"


FEATURE_NAMES = {
    "hours_studied": "Study Hours",
    "attendance_percent": "Attendance",
    "previous_scores": "Previous Performance"
}


# ============================================================
# LOAD DATASET STATISTICS
# ============================================================

df = pd.read_csv(DATA_PATH)


TRAINING_STATISTICS = {

    "hours_studied": {
        "min": float(df["hours_studied"].min()),
        "max": float(df["hours_studied"].max()),
        "mean": float(df["hours_studied"].mean()),
        "median": float(df["hours_studied"].median())
    },

    "attendance_percent": {
        "min": float(df["attendance_percent"].min()),
        "max": float(df["attendance_percent"].max()),
        "mean": float(df["attendance_percent"].mean()),
        "median": float(df["attendance_percent"].median())
    },

    "previous_scores": {
        "min": float(df["previous_scores"].min()),
        "max": float(df["previous_scores"].max()),
        "mean": float(df["previous_scores"].mean()),
        "median": float(df["previous_scores"].median())
    }
}


# ============================================================
# FORMAT FEATURE NAME
# ============================================================

def get_feature_name(feature):

    return FEATURE_NAMES.get(
        feature,
        feature
    )


# ============================================================
# CREATE SHAP EXPLANATIONS
# ============================================================

def generate_shap_explanations(
    shap_explanations
):

    positive_factors = []
    negative_factors = []

    for item in shap_explanations:

        feature = item["feature"]

        shap_value = float(
            item["shap_value"]
        )

        feature_name = get_feature_name(
            feature
        )

        contribution = abs(
            shap_value
        )

        if shap_value > 0:

            positive_factors.append({
                "feature": feature_name,
                "value": shap_value,
                "message":
                    f"{feature_name} is "
                    f"contributing positively "
                    f"by approximately "
                    f"{contribution:.2f} "
                    f"score units."
            })

        elif shap_value < 0:

            negative_factors.append({
                "feature": feature_name,
                "value": shap_value,
                "message":
                    f"{feature_name} is "
                    f"contributing negatively "
                    f"by approximately "
                    f"{contribution:.2f} "
                    f"score units."
            })

    return (
        positive_factors,
        negative_factors
    )


# ============================================================
# INPUT RANGE CHECK
# ============================================================

def check_training_range(
    study_hours,
    attendance,
    previous_score
):

    values = {

        "hours_studied":
            study_hours,

        "attendance_percent":
            attendance,

        "previous_scores":
            previous_score
    }

    warnings = []

    for feature, value in values.items():

        minimum = TRAINING_STATISTICS[
            feature
        ]["min"]

        maximum = TRAINING_STATISTICS[
            feature
        ]["max"]

        feature_name = get_feature_name(
            feature
        )

        if value < minimum or value > maximum:

            warnings.append(
                f"{feature_name} is outside "
                f"the range observed in the "
                f"training dataset "
                f"({minimum:.1f} to {maximum:.1f}). "
                f"The prediction may be less reliable "
                f"for this input."
            )

    return warnings


# ============================================================
# GENERATE STUDY GUIDANCE
# ============================================================

def generate_recommendations(
    study_hours,
    attendance,
    previous_score,
    negative_factors
):

    recommendations = []


    # --------------------------------------------------------
    # Study hours
    # --------------------------------------------------------

    study_median = TRAINING_STATISTICS[
        "hours_studied"
    ]["median"]

    if study_hours < study_median:

        recommendations.append(
            "📚 Consider increasing daily "
            "study time gradually and maintaining "
            "a consistent study schedule."
        )


    # --------------------------------------------------------
    # Attendance
    # --------------------------------------------------------

    if attendance < 75:

        recommendations.append(
            "🟡 Attendance is relatively low. "
            "Improving regular attendance may "
            "support academic performance."
        )

    elif attendance < 85:

        recommendations.append(
            "🟡 Attendance is moderate. "
            "Maintaining regular attendance "
            "could support better performance."
        )


    # --------------------------------------------------------
    # Previous performance
    # --------------------------------------------------------

    previous_median = TRAINING_STATISTICS[
        "previous_scores"
    ]["median"]

    if previous_score < previous_median:

        recommendations.append(
            "📖 Previous performance is below "
            "the dataset median. Focus on weak "
            "subjects and revise previous topics."
        )


    # --------------------------------------------------------
    # SHAP-based guidance
    # --------------------------------------------------------

    for factor in negative_factors:

        feature = factor["feature"]

        if feature == "Study Hours":

            recommendations.append(
                "🎯 Study Hours are a major "
                "negative contributor for this "
                "prediction. A consistent study "
                "routine may be useful."
            )

        elif feature == "Attendance":

            recommendations.append(
                "🎯 Attendance is negatively "
                "contributing to this prediction. "
                "Try to maintain more regular "
                "attendance."
            )

        elif feature == "Previous Performance":

            recommendations.append(
                "🎯 Previous Performance is "
                "negatively contributing to the "
                "prediction. Additional revision "
                "and practice may help."
            )


    # Remove duplicate recommendations
    unique_recommendations = []

    for recommendation in recommendations:

        if recommendation not in unique_recommendations:

            unique_recommendations.append(
                recommendation
            )

    return unique_recommendations


# ============================================================
# PREDICTION LEVEL
# ============================================================

def get_prediction_indicator(
    prediction
):

    # These are intentionally indicators,
    # not claimed official academic grades.

    if prediction < 25:

        return {
            "level": "LOW",
            "symbol": "🔴",
            "message":
                "The predicted score is relatively low "
                "on the dataset's score scale."
        }

    elif prediction < 35:

        return {
            "level": "MODERATE",
            "symbol": "🟡",
            "message":
                "The predicted score is in a moderate "
                "range on the dataset's score scale."
        }

    else:

        return {
            "level": "HIGHER",
            "symbol": "🟢",
            "message":
                "The predicted score is in a higher "
                "range on the dataset's score scale."
        }


# ============================================================
# COMPLETE EXPLANATION
# ============================================================

def build_explanation(
    prediction_result
):

    prediction = prediction_result[
        "prediction"
    ]

    study_hours = prediction_result[
        "study_hours"
    ]

    attendance = prediction_result[
        "attendance"
    ]

    previous_score = prediction_result[
        "previous_score"
    ]

    shap_explanations = prediction_result[
        "shap_explanations"
    ]


    # SHAP explanations

    (
        positive_factors,
        negative_factors
    ) = generate_shap_explanations(
        shap_explanations
    )


    # Range warnings

    range_warnings = check_training_range(
        study_hours,
        attendance,
        previous_score
    )


    # Recommendations

    recommendations = generate_recommendations(
        study_hours,
        attendance,
        previous_score,
        negative_factors
    )


    # Prediction indicator

    indicator = get_prediction_indicator(
        prediction
    )


    return {

        "prediction": prediction,

        "prediction_indicator": indicator,

        "positive_factors": positive_factors,

        "negative_factors": negative_factors,

        "range_warnings": range_warnings,

        "recommendations": recommendations
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Import our SHAP engine

    from shap_prediction import (
        explain_prediction
    )


    print("=" * 70)
    print(
        "STUDENT PERFORMANCE - "
        "EXPLANATION ENGINE"
    )
    print("=" * 70)


    # Same test student

    study_time = "3.45"
    attendance = 85
    previous_score = 70


    # Generate prediction + SHAP

    result = explain_prediction(
        study_time,
        attendance,
        previous_score
    )


    # Build explanation

    explanation = build_explanation(
        result
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    print("\nPREDICTION")
    print("-" * 70)

    print(
        f"Predicted Exam Score: "
        f"{explanation['prediction']:.2f}"
    )

    indicator = explanation[
        "prediction_indicator"
    ]

    print(
        f"{indicator['symbol']} "
        f"{indicator['level']}"
    )

    print(
        indicator["message"]
    )


    # ========================================================
    # POSITIVE FACTORS
    # ========================================================

    print("\nPOSITIVE FACTORS")
    print("-" * 70)

    if explanation["positive_factors"]:

        for factor in explanation[
            "positive_factors"
        ]:

            print(
                "🟢 " +
                factor["message"]
            )

    else:

        print(
            "No significant positive "
            "SHAP contributions."
        )


    # ========================================================
    # NEGATIVE FACTORS
    # ========================================================

    print("\nNEGATIVE FACTORS")
    print("-" * 70)

    if explanation["negative_factors"]:

        for factor in explanation[
            "negative_factors"
        ]:

            print(
                "🔴 " +
                factor["message"]
            )

    else:

        print(
            "No significant negative "
            "SHAP contributions."
        )


    # ========================================================
    # RANGE WARNINGS
    # ========================================================

    print("\nDATA RANGE WARNINGS")
    print("-" * 70)

    if explanation["range_warnings"]:

        for warning in explanation[
            "range_warnings"
        ]:

            print(
                "⚠️ " + warning
            )

    else:

        print(
            "No input is outside the "
            "training-data range."
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    print("\nRECOMMENDATIONS")
    print("-" * 70)

    if explanation["recommendations"]:

        for recommendation in explanation[
            "recommendations"
        ]:

            print(
                recommendation
            )

    else:

        print(
            "No additional recommendations."
        )


    print("\n" + "=" * 70)
    print("EXPLANATION ENGINE TEST COMPLETE")
    print("=" * 70)