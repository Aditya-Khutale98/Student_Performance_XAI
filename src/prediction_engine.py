import os
import sys
import pandas as pd
import joblib


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
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/student_performance_bounded.pkl"
)

FEATURES = [
    "Study_Hours_Daily",
    "Attendance",
    "Previous_Scores"
]


# ============================================================
# LOAD FINAL BOUNDED MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# STUDY TIME CONVERSION
# ============================================================

def convert_study_time(time_value):
    """
    Convert study time entered as HH.MM into decimal hours.

    Examples:

        3.45 -> 3 hours 45 minutes -> 3.75 hours
        2.30 -> 2 hours 30 minutes -> 2.50 hours
        5    -> 5 hours -> 5.00 hours
        24.00 -> 24 hours -> 24.00 hours

    IMPORTANT:
    3.45 does NOT mean 3.45 decimal hours.
    It means 3 hours 45 minutes.
    """

    time_text = str(
        time_value
    ).strip()


    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not time_text:

        raise ValueError(
            "Study time cannot be empty."
        )


    # --------------------------------------------------------
    # Handle HH.MM format
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
    # Handle whole-hour input
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
    # Validate minutes
    # --------------------------------------------------------

    if minutes < 0 or minutes >= 60:

        raise ValueError(
            "Minutes must be between 00 and 59."
        )


    # --------------------------------------------------------
    # Validate hours
    # --------------------------------------------------------

    if hours < 0:

        raise ValueError(
            "Hours cannot be negative."
        )


    if hours > 24:

        raise ValueError(
            "Study time cannot be more than 24 hours per day."
        )


    # --------------------------------------------------------
    # Special case:
    # 24.01, 24.30 etc. are invalid
    # --------------------------------------------------------

    if hours == 24 and minutes > 0:

        raise ValueError(
            "Study time cannot be greater than 24 hours."
        )


    # --------------------------------------------------------
    # Convert HH.MM to decimal hours
    # --------------------------------------------------------

    decimal_hours = (
        hours + minutes / 60.0
    )


    return decimal_hours


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs(
    study_time,
    attendance_percent,
    previous_scores
):
    """
    Validate all user inputs before prediction.
    """

    # --------------------------------------------------------
    # Study time
    # --------------------------------------------------------

    decimal_hours = convert_study_time(
        study_time
    )


    if decimal_hours < 0 or decimal_hours > 24:

        raise ValueError(
            "Study time must be between 0 and 24 hours per day."
        )


    # --------------------------------------------------------
    # Attendance
    # --------------------------------------------------------

    try:

        attendance = float(
            attendance_percent
        )

    except (TypeError, ValueError):

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

    except (TypeError, ValueError):

        raise ValueError(
            "Previous performance must be a number."
        )


    if previous_score < 0 or previous_score > 100:

        raise ValueError(
            "Previous performance must be between 0 and 100."
        )


    return (
        decimal_hours,
        attendance,
        previous_score
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_exam_score(
    study_time,
    attendance_percent,
    previous_scores
):
    """
    Generate an exam-score prediction using the
    final Logistic Bounded Regression model.
    """

    # --------------------------------------------------------
    # Validate inputs
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
    # Create model input
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            decimal_hours,
            attendance,
            previous_score
        ]],
        columns=FEATURES
    )


    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # --------------------------------------------------------
    # Final safety bound
    # --------------------------------------------------------

    prediction = max(
        0.0,
        min(
            100.0,
            float(prediction)
        )
    )


    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {

        "study_time_input":
            str(study_time),

        "study_hours_decimal":
            float(decimal_hours),

        "attendance_percent":
            float(attendance),

        "previous_scores":
            float(previous_score),

        "predicted_exam_score":
            float(prediction)
    }


# ============================================================
# TEST THE ENGINE
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 75
    )

    print(
        "STUDENT PERFORMANCE "
        "PREDICTION ENGINE"
    )

    print(
        "=" * 75
    )


    # --------------------------------------------------------
    # Example student
    # --------------------------------------------------------

    study_time = "3.45"

    attendance = 85

    previous_score = 70


    print("\nINPUT")

    print(
        "-" * 75
    )

    print(
        f"Study Time       : {study_time}"
    )

    print(
        f"Attendance       : {attendance}%"
    )

    print(
        f"Previous Score   : {previous_score}"
    )


    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    try:

        result = predict_exam_score(
            study_time,
            attendance,
            previous_score
        )


        # ----------------------------------------------------
        # Processed input
        # ----------------------------------------------------

        print("\nPROCESSED INPUT")

        print(
            "-" * 75
        )

        print(
            f"Study Hours      : "
            f"{result['study_hours_decimal']:.2f}"
        )

        print(
            f"Attendance       : "
            f"{result['attendance_percent']:.2f}%"
        )

        print(
            f"Previous Score   : "
            f"{result['previous_scores']:.2f}"
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        print("\nPREDICTION")

        print(
            "-" * 75
        )

        print(
            f"Predicted Exam Score : "
            f"{result['predicted_exam_score']:.2f}"
        )


        # ----------------------------------------------------
        # Range check
        # ----------------------------------------------------

        prediction = (
            result[
                "predicted_exam_score"
            ]
        )


        print("\nRANGE CHECK")

        print(
            "-" * 75
        )

        print(
            f"Prediction range : "
            f"{prediction:.2f}"
        )


        if 0 <= prediction <= 100:

            print(
                "PASS: Prediction is within 0-100."
            )

        else:

            print(
                "FAIL: Prediction is outside 0-100."
            )


    except ValueError as error:

        print("\nINPUT ERROR")

        print(
            "-" * 75
        )

        print(
            error
        )


    print(
        "\n" + "=" * 75
    )

    print(
        "PREDICTION ENGINE TEST COMPLETE"
    )

    print(
        "=" * 75
    )