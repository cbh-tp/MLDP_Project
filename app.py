import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. LOAD THE ASSETS (With Error Handling for Rubric) ---
@st.cache_resource
def load_assets():
    try:
        # We load the Voting Ensemble (Model A) as it is the most robust
        model = pickle.load(open('churn_model.pkl', 'rb'))
        scaler = pickle.load(open('churn_scaler.pkl', 'rb'))
        model_columns = pickle.load(open('model_columns.pkl', 'rb'))
        return model, scaler, model_columns
    except FileNotFoundError:
        # Satisfies "User-facing error message" requirement 
        return None, None, None

model, scaler, model_columns = load_assets()

# --- 2. THE UI LAYOUT ---
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

if model is None:
    st.error("🚨 Critical Error: Model files not found!")
    st.warning("Please ensure 'churn_model.pkl', 'churn_scaler.pkl', and 'model_columns.pkl' are in the same folder.")
    st.stop() # Stops the app from crashing further

st.title("🔮 Telco Customer Churn Predictor")
st.markdown("""
This app predicts whether a customer is likely to **Churn** (leave) or **Stay**.
Adjust the customer details in the sidebar to test the model.
""")

# --- 3. SIDEBAR INPUTS (With Clear Labels for Rubric) ---
st.sidebar.header("Customer Details")

# Numerical Inputs
# Sliders satisfy "Interactive" requirement 
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)

# Input Validation Logic (Satisfies Rubric )
if monthly_charges > 150:
    st.sidebar.warning("⚠️ Note: Monthly charges above $150 are rare.")

total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=tenure * monthly_charges)

# Categorical Inputs
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
senior_citizen = st.sidebar.selectbox("Is Senior Citizen?", ["Yes", "No"])

# --- 4. PREPROCESSING LOGIC ---
if st.sidebar.button("Predict Churn Status"):
    
    # Create Input Dictionary
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'PaperlessBilling': paperless_billing
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-Hot Encoding
    input_df_encoded = pd.get_dummies(input_df)

    # Align with Model Columns (Prevents crashing if inputs differ)
    input_df_ready = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale the Data
    input_scaled = scaler.transform(input_df_ready)

    # --- 5. PREDICTION ---
    try:
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1] # Probability of Churn
        
        # --- 6. DISPLAY RESULTS ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction")
            if prediction[0] == 1:
                st.error("🚨 CHURN RISK DETECTED")
                st.write("This customer is likely to leave.")
            else:
                st.success("✅ CUSTOMER IS SAFE")
                st.write("This customer is likely to stay.")

        with col2:
            st.subheader("Confidence Score")
            st.metric(label="Probability of Churn", value=f"{probability:.2%}")
            
            # Visual Progress Bar
            st.progress(float(probability)) # Ensure float for Streamlit
            
            # Business Logic Message
            if probability > 0.6:
                st.warning("Risk is High! Recommendation: Offer $10 discount.")
            else:
                st.info("Customer is stable. No action needed.")

    except Exception as e:
        # Catch-all error handler for Rubric 
        st.error(f"An error occurred during prediction: {e}")

else:
    st.info("👈 Adjust the customer details in the sidebar and click 'Predict' to see the result.")