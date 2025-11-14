# ===============================
# Streamlit Dashboard for Dropout Prediction
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1️⃣ Load trained model
with open("best_catboost_model_final.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Student Dropout Dashboard", layout="centered")
st.title("📊 Student Dropout Prediction Dashboard")
st.markdown("Enter student information to predict Dropout or Graduate. Support recommendations are provided for Dropouts.")

# 2️⃣ Input features with realistic ranges from dataset
prev_qual_grade = st.number_input(
    "Previous qualification (grade)", 
    min_value=95.0, max_value=190.0, value=132.6, step=1.0
)
admission_grade = st.number_input(
    "Admission grade", 
    min_value=95.0, max_value=190.0, value=126.9, step=1.0
)
age = st.number_input(
    "Age at enrollment", 
    min_value=17, max_value=70, value=20, step=1
)
cu_1st_approved = st.number_input(
    "Curricular units 1st sem (approved)", 
    min_value=0, max_value=26, value=5, step=1
)
cu_1st_grade = st.number_input(
    "Curricular units 1st sem (grade)", 
    min_value=0.0, max_value=18.875, value=12.28, step=0.01
)
cu_2nd_approved = st.number_input(
    "Curricular units 2nd sem (approved)", 
    min_value=0, max_value=20, value=5, step=1
)
cu_2nd_grade = st.number_input(
    "Curricular units 2nd sem (grade)", 
    min_value=0.0, max_value=18.571, value=12.2, step=0.01
)

# 3️⃣ Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[prev_qual_grade, admission_grade, age,
                                cu_1st_approved, cu_1st_grade,
                                cu_2nd_approved, cu_2nd_grade]],
                              columns=[
                                  'Previous qualification (grade)',
                                  'Admission grade',
                                  'Age at enrollment',
                                  'Curricular units 1st sem (approved)',
                                  'Curricular units 1st sem (grade)',
                                  'Curricular units 2nd sem (approved)',
                                  'Curricular units 2nd sem (grade)'
                              ])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 0:
        st.error("⚠️ Predicted: Dropout")
        st.markdown("""
        **Recommended Support for Dropouts:**
        - Academic counseling
        - Tutoring for difficult subjects
        - Time management workshops
        - Peer mentoring
        - Regular progress monitoring
        """)
    else:
        st.success("✅ Predicted: Graduate")
        st.balloons()  # 🎈 Celebration effect
