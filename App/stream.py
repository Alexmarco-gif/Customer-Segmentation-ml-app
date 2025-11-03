import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🎯 Load Saved Models
model = joblib.load('../Model/random_forest_model.pkl')
scaler = joblib.load('../Model/scaler.pkl')
features = joblib.load('../Model/features.pkl')

# 🌈 Page Configuration & Styling
st.set_page_config(page_title="Customer Segmentation App", page_icon="🧠", layout="centered")

# Add custom CSS for background color and card styling
st.markdown("""
    <style>
        body {
            background-color: #F8F9FA;
        }
        .main {
            background-color: #FFFFFF;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #333333;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# 🧭 Sidebar Section
st.sidebar.title("📘 About the Project")
st.sidebar.markdown("""
This project demonstrates an **end-to-end Machine Learning pipeline** for **Customer Segmentation**.

### 🚀 Features
- Unsupervised clustering with K-Means  
- Supervised classification using XGBoost / Random Forest  
- Streamlit app for real-time prediction  
- Scalable model deployment setup  

### 🧠 Goal
Help businesses understand their customers better and **personalize marketing strategies**.

Developed with ❤️ using **Python, Scikit-learn, Streamlit, and Joblib**.
""")

# 🧩 App Title
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🧠 Customer Segmentation Prediction App")
st.markdown("Predict which **customer segment** a new customer belongs to based on demographic and spending behavior.")

# 🗂 Tabs Layout
tab1, tab2, tab3 = st.tabs(["🧍 Customer Input", "🔮 Prediction", "📊 Model Info"])

# 🧍 Tab 1: Customer Input
with tab1:
    st.subheader("Enter Customer Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Annual Income ($)", min_value=10, max_value=150, value=50)
    spending_score = st.slider("Spending Score (1–100)", 1, 100, 50)

    # Feature engineering
    spending_efficiency = spending_score / (income + 1)
    income_spend_interaction = income * spending_score
    gender_encoded = 1 if gender == "Male" else 0

    # Create dataframe
    input_data = pd.DataFrame({
        "Gender": [gender_encoded],
        "Age": [age],
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending_score],
        "Spending_Efficiency": [spending_efficiency],
        "Income_Spend_Interaction": [income_spend_interaction]
    })
    input_data = input_data[features]

    st.write("**Preview of Input Data:**")
    st.dataframe(input_data, use_container_width=True)

# 🔮 Tab 2: Prediction
with tab2:
    st.subheader("Customer Segment Prediction")

    if st.button("Predict Customer Segment"):
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        
        st.success(f"✅ Predicted Customer Segment: **Segment {prediction}**")

        # Interpretation guide
        st.markdown("""
        ### 📘 Segment Interpretation (Example)
        - **Segment 0** → Low-income, low-spending customers  
        - **Segment 1** → Mid-income, average-spending customers  
        - **Segment 2** → High-income, luxury-spending customers  
        - **Segment 3** → Young, high-spending customers     
        - **Segment 4** → Senior, budget-conscious customers
        """)

# 📊 Tab 3: Model Info
with tab3:
    st.subheader("Model Details")
    st.markdown("""
    **Model Pipeline Summary**
    - **Clustering Algorithm:** K-Means  
    - **Classifier:** RandomForest  
    - **Scaling:** StandardScaler  
    - **Storage:** Joblib (.pkl files)  

    **Feature Set Used:**
    - Gender (encoded)  
    - Age  
    - Annual Income (k$)  
    - Spending Score (1-100)  
    - Spending Efficiency  
    - Income-Spend Interaction  

    ---
    **Deployment Tools**
    - Streamlit for UI  
    - Joblib for model persistence  
    - Scikit-learn for ML algorithms  
    """)

st.markdown("</div>", unsafe_allow_html=True)