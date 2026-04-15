import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("Best_Model.pkl")

model = load_model()

# ---------------- HEADER ---------------- #
st.title("📊 Customer Churn Prediction System")
st.markdown("### AI-powered Customer Retention Insights")

st.markdown("""
This application predicts whether a customer is likely to churn based on behavioral and service data.
""")

st.divider()

# ---------------- INPUT SECTION ---------------- #
st.subheader("🧾 Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col3:
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=500.0)
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

st.divider()

# ---------------- ADVANCED OPTIONS ---------------- #
with st.expander("⚙️ Advanced Service Details"):

    col4, col5 = st.columns(2)

    with col4:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col5:
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

st.divider()

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict Churn Risk", use_container_width=True):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📢 Prediction Result")

        st.progress(min(int(probability * 100), 100))

        colA, colB = st.columns(2)

        with colA:
            st.metric("Churn Probability", f"{probability:.2%}")

        with colB:
            st.metric(
                "Risk Level",
                "High" if probability > 0.7
                else "Medium" if probability > 0.4
                else "Low"
            )

        st.divider()

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")

            if probability > 0.75:
                st.warning("🔥 Very High Risk — Immediate retention action required")

            elif probability > 0.5:
                st.info("⚡ Moderate Risk — Monitor and engage customer")

        else:
            st.success("✅ Customer is likely to stay")
            st.info("👍 Stable customer — maintain engagement")

        st.divider()

        # ---------------- FEATURE IMPORTANCE ---------------- #

        st.subheader("🔍 Key Factors Influencing Prediction")

        try:
            model_step = model.named_steps["model"]
            preprocess = model.named_steps["preprocess"]

            feature_names = preprocess.get_feature_names_out()
            importance = model_step.feature_importances_

            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            top_features = imp_df.head(5)

            st.write("Top factors affecting churn:")

            for i, row in top_features.iterrows():
                st.write(f"• {row['Feature']}")

            st.bar_chart(top_features.set_index("Feature"))

        except:
            st.info("Feature importance not available for this model.")

    except Exception as e:
        st.error("❌ Prediction failed. Check model compatibility.")
        st.exception(e)

# ---------------- FOOTER ---------------- #
st.divider()

st.markdown("""
---
📌 **Project:** Customer Churn Prediction  
🤖 Built using Machine Learning & Streamlit  
⚠️ Predictions are based on historical data and may not be 100% accurate.
""")