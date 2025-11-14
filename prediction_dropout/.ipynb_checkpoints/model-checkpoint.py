# task6_dashboard.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load your trained CatBoost model ---
with open("best_catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Load feature importances if available ---
feature_importances = pd.DataFrame({
    "Feature": ["Curricular units 1st sem (grade)", "Admission grade",
                "Previous qualification (grade)", "Age at enrollment"],
    "Importance": [53.85, 12.54, 11.33, 22.28]  # replace with actual values if saved
}).sort_values(by="Importance", ascending=False)

# --- App Title ---
st.title("🎓 Student Dropout Prediction Dashboard")
st.write("""
This dashboard predicts whether a student is likely to **Dropout**, **Enroll**, or **Graduate** 
based on their academic performance and enrollment information.
""")

# --- Input Form ---
st.header("Enter Student Information")

curricular_units = st.number_input(
    "Curricular units 1st semester (grade)", min_value=0.0, max_value=20.0, value=10.0, step=0.1
)
admission_grade = st.number_input(
    "Admission grade", min_value=50.0, max_value=200.0, value=120.0, step=0.1
)
previous_qualification = st.number_input(
    "Previous qualification (grade)", min_value=50.0, max_value=200.0, value=130.0, step=0.1
)
age_at_enrollment = st.number_input(
    "Age at enrollment", min_value=17, max_value=40, value=20
)

# --- Create Input DataFrame ---
input_data = pd.DataFrame({
    "Curricular units 1st sem (grade)": [curricular_units],
    "Admission grade": [admission_grade],
    "Previous qualification (grade)": [previous_qualification],
    "Age at enrollment": [age_at_enrollment]
})

# --- Suggested Interventions ---
intervention_map = {
    "Dropout": [
        "Enroll in Academic Support Program",
        "Schedule Counseling Session",
        "Join Tutoring Workshops"
    ],
    "Enrolled": ["Continue Current Path", "Monitor Progress"],
    "Graduate": ["Prepare for Graduation", "Career Counseling"]
}

# --- Prediction ---
if st.button("Predict Dropout Risk"):
    # Model prediction
    prediction = model.predict(input_data)
    if isinstance(prediction, (np.ndarray, list)):
        prediction = prediction[0]
        if isinstance(prediction, (np.ndarray, list)):
            prediction = prediction.item()

    prediction_proba = model.predict_proba(input_data)[0]  # probabilities for each class

    # Map numeric target to labels
    target_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    predicted_label = target_map[int(prediction)]

    # --- Display Results ---
    st.subheader("Prediction Result")
    if predicted_label == "Dropout":
        st.error(f"⚠️ Predicted Status: {predicted_label}")
    elif predicted_label == "Enrolled":
        st.warning(f"Predicted Status: {predicted_label}")
    else:
        st.success(f"Predicted Status: {predicted_label}")

    # --- Risk Probabilities ---
    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame({
        "Status": ["Dropout", "Enrolled", "Graduate"],
        "Probability": prediction_proba
    })
    st.bar_chart(proba_df.set_index("Status"))

    # --- Dropout Interventions ---
    if predicted_label == "Dropout":
        st.subheader("💡 Suggested Programs/Interventions")
        for program in intervention_map[predicted_label]:
            st.markdown(f"- {program}")

    # --- Risk Warning ---
    dropout_prob = prediction_proba[0]
    if dropout_prob > 0.6:
        st.warning(f"⚠️ High risk of dropout: {dropout_prob*100:.1f}%")

    # --- Show Feature Importance ---
    st.subheader("📊 Feature Importances")
    st.bar_chart(feature_importances.set_index("Feature"))

# --- Optional Batch Prediction ---
st.header("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV file with students data", type="csv")
if uploaded_file is not None:
    df_batch = pd.read_csv(uploaded_file)
    predictions = model.predict(df_batch)
    df_batch['Predicted Status'] = [target_map[int(p)] for p in predictions]
    st.dataframe(df_batch)
    df_batch.to_csv("batch_predictions_with_interventions.csv", index=False)
    st.success("✅ Predictions saved!")
    
st.info("💡 Adjust the values above and click **Predict Dropout Risk** to update the prediction.")
