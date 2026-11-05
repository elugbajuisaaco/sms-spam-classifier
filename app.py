"""Streamlit interface for the reusable SMS spam prediction engine."""

import logging

import streamlit as st

from src.predict import predict_message


LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📨",
    layout="centered",
)

st.title("SMS Spam Classifier")
st.write(
    "Paste a text message below to estimate whether it is spam or legitimate."
)

message = st.text_area(
    "SMS message",
    placeholder="Type or paste an SMS message here...",
    height=180,
    key="sms_message",
)

if st.button("Classify Message", type="primary"):
    st.session_state.pop("prediction_result", None)
    try:
        result = predict_message(message)
    except ValueError as error:
        if not message.strip():
            st.session_state["prediction_error"] = (
                "Please enter an SMS message before classifying."
            )
        else:
            LOGGER.exception("Prediction failed with a value error.")
            st.session_state["prediction_error"] = (
                "Sorry, the message could not be classified. Please try again."
            )
    except Exception:
        LOGGER.exception("Unexpected error while classifying an SMS message.")
        st.session_state["prediction_error"] = (
            "Sorry, the message could not be classified. Please try again."
        )
    else:
        st.session_state.pop("prediction_error", None)
        st.session_state["prediction_result"] = {
            "prediction": result.prediction,
            "spam_probability": result.spam_probability,
        }

if error_message := st.session_state.get("prediction_error"):
    st.error(error_message)

if result := st.session_state.get("prediction_result"):
    if result["prediction"] == "spam":
        st.error("## SPAM")
    else:
        st.success("## HAM")

    st.metric("Estimated spam probability", f"{result['spam_probability']:.2%}")
    st.caption(
        "This probability is the model's estimate, not a guarantee that the "
        "message is spam."
    )