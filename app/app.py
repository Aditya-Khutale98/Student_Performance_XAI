import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORT MASTER SYSTEM
# ============================================================

from src.student_performance_system import (
    predict_student_performance
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title=
        "Student Performance Predictor",

    page_icon=
        "🎓",

    layout=
        "centered"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎓 Student Performance Predictor"
)

st.write(
    "Predict exam performance and understand "
    "which factors influenced the prediction."
)


st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader(
    "📝 Enter Student Information"
)


st.info(
    "For Study Hours, enter time as HH.MM. "
    "For example, 3.45 means 3 hours 45 minutes."
)


# ------------------------------------------------------------
# Study Hours
# ------------------------------------------------------------

study_time = st.text_input(

    "Study Hours per Day",

    value="3.45",

    help=(
        "Enter study time as HH.MM. "
        "Example: 3.45 = 3 hours 45 minutes. "
        "Maximum: 24.00 hours."
    )
)


# ------------------------------------------------------------
# Attendance
# ------------------------------------------------------------

attendance = st.number_input(

    "Attendance (%)",

    min_value=0.0,

    max_value=100.0,

    value=85.0,

    step=1.0
)


# ------------------------------------------------------------
# Previous Performance
# ------------------------------------------------------------

previous_score = st.number_input(

    "Previous Performance",

    min_value=0.0,

    max_value=100.0,

    value=70.0,

    step=1.0
)


st.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(

    "🔮 Predict Performance",

    type="primary",

    use_container_width=True
)


# ============================================================
# RUN PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Run complete backend system
        # ----------------------------------------------------

        result = predict_student_performance(

            study_time=
                study_time,

            attendance_percent=
                attendance,

            previous_scores=
                previous_score
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        st.subheader(
            "📊 Prediction"
        )


        prediction = result[
            "prediction"
        ]


        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(

                "Predicted Exam Score",

                f"{prediction['score']:.2f} / 100"
            )


        with col2:

            st.metric(

                "Typical Validation Error",

                f"±{result['uncertainty']['typical_mae']:.2f}"
            )


        # ----------------------------------------------------
        # Prediction level
        # ----------------------------------------------------

        st.write(

            f"{prediction['symbol']} "
            f"**{prediction['level']}**"
        )


        st.caption(
            prediction["message"]
        )


        # ====================================================
        # PREDICTION UNCERTAINTY
        # ====================================================

        st.subheader(
            "⚠️ Prediction Uncertainty"
        )


        lower = result[
            "uncertainty"
        ][
            "lower_bound"
        ]


        upper = result[
            "uncertainty"
        ][
            "upper_bound"
        ]


        st.info(

            f"Empirical error range: "
            f"**{lower:.2f} – {upper:.2f}**"
        )


        st.caption(

            result[
                "uncertainty"
            ][
                "message"
            ]
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.subheader(
            "🔍 Why did the model make this prediction?"
        )


        st.write(

            "SHAP values show how each input "
            "contributed to this particular "
            "prediction relative to the model's "
            "reference data."
        )


        # ----------------------------------------------------
        # SHAP base value
        # ----------------------------------------------------

        base_value = result[
            "shap"
        ][
            "base_value"
        ]


        st.caption(

            f"Model reference value: "
            f"{base_value:.2f}"
        )


        # ----------------------------------------------------
        # SHAP contributions
        # ----------------------------------------------------

        contributions = result[
            "shap"
        ][
            "contributions"
        ]


        for item in contributions:

            feature_name = item[
                "feature_name"
            ]


            shap_value = item[
                "shap_value"
            ]


            if shap_value > 0:

                st.success(

                    f"🟢 **{feature_name}** "
                    f"contributed "
                    f"**+{shap_value:.2f}** "
                    f"score points."
                )


            elif shap_value < 0:

                st.error(

                    f"🔴 **{feature_name}** "
                    f"contributed "
                    f"**{shap_value:.2f}** "
                    f"score points."
                )


            else:

                st.info(

                    f"⚪ **{feature_name}** "
                    f"had almost no contribution."
                )


        # ====================================================
        # DATA COVERAGE / EXTRAPOLATION
        # ====================================================

        coverage = result[
            "data_coverage"
        ]


        if coverage:

            st.subheader(
                "⚠️ Training Data Coverage"
            )


            st.write(

                "Your input is valid, but one or more "
                "values are outside the ranges observed "
                "in the training dataset."
            )


            for item in coverage:

                st.warning(
                    item["message"]
                )


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.subheader(
            "💡 Recommendations"
        )


        recommendations = result[
            "recommendations"
        ]


        if recommendations:

            for recommendation in recommendations:

                st.write(

                    f"• {recommendation}"
                )


        else:

            st.write(
                "No additional recommendations."
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        with st.expander(
            "📋 View Processed Input"
        ):

            st.write(

                f"**Study Time Entered:** "
                f"{result['input']['study_time']}"
            )


            st.write(

                f"**Study Hours Used by Model:** "
                f"{result['input']['study_hours']:.2f} "
                f"hours/day"
            )


            st.write(

                f"**Attendance:** "
                f"{result['input']['attendance']:.2f}%"
            )


            st.write(

                f"**Previous Performance:** "
                f"{result['input']['previous_score']:.2f}"
            )


        # ====================================================
        # SHAP NUMERICAL DETAILS
        # ====================================================

        with st.expander(
            "📈 View SHAP Numerical Details"
        ):

            for item in contributions:

                st.write(

                    f"**{item['feature_name']}**: "
                    f"{item['shap_value']:+.4f}"
                )


            st.write(

                f"**Base Value:** "
                f"{base_value:.4f}"
            )


            st.write(

                f"**Final Prediction:** "
                f"{prediction['score']:.4f}"
            )


        # ====================================================
        # SYSTEM SUCCESS
        # ====================================================

        st.success(
            "Prediction and explanation generated successfully."
        )


    # ========================================================
    # INPUT ERROR
    # ========================================================

    except ValueError as error:

        st.error(
            f"❌ Input Error: {error}"
        )


    # ========================================================
    # UNEXPECTED ERROR
    # ========================================================

    except Exception as error:

        st.error(
            "❌ An unexpected error occurred."
        )

        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Student Performance Prediction + Explainable AI"
)


st.caption(
    "Predictions are model-based estimates and "
    "should not be treated as guaranteed results."
)