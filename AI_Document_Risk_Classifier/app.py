
import streamlit as st
import joblib
import pandas as pd
import os
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="AI Document Risk Classifier",
    page_icon="📄",
    layout="centered"
)

# ----------------------------
# File Paths and Model Loading
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model")
HISTORY_FILE = os.path.join(BASE_DIR, "history.csv")

try:
    model = joblib.load(os.path.join(MODEL_PATH, "risk_classifier.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "vectorizer.pkl"))

except FileNotFoundError:
    st.error("Model files not found! Check your model folder.")
    st.stop()

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("📄 AI Document Risk Classifier")

st.sidebar.markdown("---")

st.sidebar.header("Project Details")

st.sidebar.write("**Project:** AI Document Risk & Compliance Classifier")
st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Vectorizer:** TF-IDF")
st.sidebar.write("**Language:** Python")
st.sidebar.write("**Framework:** Streamlit")

st.sidebar.markdown("---")

st.sidebar.header("Features")

st.sidebar.write("✅ Text Classification")
st.sidebar.write("✅ PDF Upload")
st.sidebar.write("✅ TXT Upload")
st.sidebar.write("✅ Dashboard")
st.sidebar.write("✅ History")
st.sidebar.write("✅ CSV Export")

st.sidebar.markdown("---")

st.sidebar.success("Project Ready ✅")

# ----------------------------
# Main Title
# ----------------------------

st.title("📄 AI Document Risk & Compliance Classifier")

st.write(
    "Classify documents as **High Risk** or **Low Risk**."
)

# ----------------------------
# Upload File
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)

user_input = ""

if uploaded_file is not None:

    if uploaded_file.name.lower().endswith(".pdf"):

        try:
            pdf = PdfReader(uploaded_file)

            extracted_text = ""

            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text:
                    extracted_text += page_text + "\n"

            user_input = extracted_text

        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    elif uploaded_file.name.lower().endswith(".txt"):

        try:
            extracted_text = uploaded_file.read().decode(
                "utf-8",
                errors="replace"
            )

            user_input = extracted_text

        except Exception as e:
            st.error(f"Error reading TXT: {e}")

    st.subheader("📄 Extracted Text")

    st.text_area(
        "Document Content",
        user_input,
        height=220
    )

else:

    user_input = st.text_area(
        "Type your document here",
        height=220
    )

# ----------------------------
# Classify Button
# ----------------------------

if st.button("🔍 Classify"):

    if user_input.strip() == "":

        st.warning("Please enter some text or upload a document.")

    else:

        # Convert text into TF-IDF features

        text = vectorizer.transform([user_input])

        # Predict risk category

        prediction = model.predict(text)

        result = prediction[0]

        # ----------------------------
        # Sensitive Keyword Detection
        # ----------------------------

        sensitive_keywords = [
            "aadhaar",
            "aadhar",
            "pan",
            "passport",
            "salary",
            "bank",
            "account",
            "credit card",
            "debit card",
            "password",
            "upi",
            "ifsc",
            "cvv",
            "otp",
            "confidential",
            "secret",
            "employee",
            "license"
        ]

        detected = []

        document_lower = user_input.lower()

        for keyword in sensitive_keywords:

            if keyword in document_lower:
                detected.append(keyword)

        # ----------------------------
        # Prediction Result
        # ----------------------------

        st.markdown("---")

        st.subheader("Prediction Result")

        if result == "High Risk":

            st.error(f"🔴 Prediction: {result}")

        else:

            st.success(f"🟢 Prediction: {result}")

        # ----------------------------
        # Explanation
        # ----------------------------

        st.subheader("Why this prediction?")

        if result == "High Risk":

            st.write("""
            This document appears to contain sensitive
            or confidential information such as:

            - Personal Information
            - Financial Data
            - Government IDs
            - Banking Details
            - Company Confidential Data

            **Recommendation:**

            Store this document securely and restrict
            unauthorized access.
            """)

        else:

            st.write("""
            This document appears to contain general
            information.

            No highly confidential information was
            classified by the model.
            """)

        # ----------------------------
        # Sensitive Keywords Result
        # ----------------------------

        st.subheader("🔍 Sensitive Keywords Detected")

        if detected:

            st.warning(", ".join(detected))

        else:

            st.info("No sensitive keywords detected.")

        # ----------------------------
        # Save History
        # ----------------------------

        history = {
            "Document": user_input[:60],
            "Prediction": result
        }

        if os.path.exists(HISTORY_FILE):

            df = pd.read_csv(HISTORY_FILE)

            df = pd.concat(
                [df, pd.DataFrame([history])],
                ignore_index=True
            )

        else:

            df = pd.DataFrame([history])

        df.to_csv(HISTORY_FILE, index=False)

        st.success("Classification saved to history!")

# ----------------------------
# Project Statistics
# ----------------------------

st.markdown("---")

st.subheader("📊 Project Statistics")

st.write("Model: Logistic Regression")
st.write("Vectorizer: TF-IDF")
st.write("Dataset Size: 100+ Documents")
st.write("Classes: High Risk / Low Risk")

# ----------------------------
# Dashboard
# ----------------------------

st.markdown("---")

st.subheader("📊 Dashboard")

if os.path.exists(HISTORY_FILE):

    dashboard_df = pd.read_csv(HISTORY_FILE)

    total_docs = len(dashboard_df)

    high_risk = len(
        dashboard_df[
            dashboard_df["Prediction"] == "High Risk"
        ]
    )

    low_risk = len(
        dashboard_df[
            dashboard_df["Prediction"] == "Low Risk"
        ]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("📄 Total", total_docs)

    col2.metric("🔴 High Risk", high_risk)

    col3.metric("🟢 Low Risk", low_risk)

    # Pie Chart

    if total_docs > 0:

        fig, ax = plt.subplots()

        ax.pie(
            [high_risk, low_risk],
            labels=["High Risk", "Low Risk"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Risk Distribution")

        st.pyplot(fig)

else:

    st.info("No classification data available yet.")

# ----------------------------
# Classification History
# ----------------------------

st.markdown("---")

st.subheader("🕒 Classification History")

if os.path.exists(HISTORY_FILE):

    history_df = pd.read_csv(HISTORY_FILE)

    st.dataframe(history_df)

    # CSV Download

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name="classification_history.csv",
        mime="text/csv"
    )

else:

    st.info("No history available.")

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: gray;
    font-size: 14px;'>
        <b>Developed by: NIKHIL JOSHI 04023815724</b><br>
        B.Tech (CSE-DS) | INTERNSHIP PROJECT
    </div>
    """,
    unsafe_allow_html=True
)
