import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("regression_model.pkl")

# Prediction function
def predict_exam_score(input_data):
    input_df = pd.DataFrame([input_data])
    input_transformed = preprocessor.transform(input_df)
    predicted_score = model.predict(input_transformed)
    return predicted_score[0]

# Streamlit UI
st.title("📊 Student Exam Score Predictor")

st.markdown("Provide student details below to predict the exam score:")

# Input fields
Hours_Studied = st.slider("Hours Studied", 0, 100, 10)
Attendance = st.slider("Attendance (%)", 0, 100, 75)
Sleep_Hours = st.slider("Sleep Hours per Night", 0, 12, 7)
Previous_Scores = st.slider("Previous Score (%)", 0, 100, 70)
Tutoring_Sessions = st.number_input("Number of Tutoring Sessions", min_value=0, max_value=20, value=2)
Physical_Activity = st.number_input("Physical Activity (hours/week)", min_value=0, max_value=20, value=3)

# Categorical inputs
Parental_Involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
Access_to_Resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
Extracurricular_Activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
Motivation_Level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
Internet_Access = st.selectbox("Internet Access", ["Yes", "No"])
Family_Income = st.selectbox("Family Income", ["Low", "Medium", "High"])
Teacher_Quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
School_Type = st.selectbox("School Type", ["Public", "Private"])
Peer_Influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
Learning_Disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
Parental_Education_Level = st.selectbox("Parental Education Level", ["High School", "Undergraduate", "Postgraduate"])
Distance_from_Home = st.selectbox("Distance from Home", ["Near", "Far"])
Gender = st.selectbox("Gender", ["Male", "Female"])

# Create input dictionary
new_student = {
    "Hours_Studied": Hours_Studied,
    "Attendance": Attendance,
    "Parental_Involvement": Parental_Involvement,
    "Access_to_Resources": Access_to_Resources,
    "Extracurricular_Activities": Extracurricular_Activities,
    "Sleep_Hours": Sleep_Hours,
    "Previous_Scores": Previous_Scores,
    "Motivation_Level": Motivation_Level,
    "Internet_Access": Internet_Access,
    "Gender": Gender,
    "Distance_from_Home": Distance_from_Home,
    "Teacher_Quality": Teacher_Quality,
    "Parental_Education_Level": Parental_Education_Level,
    "Tutoring_Sessions": Tutoring_Sessions,
    "School_Type": School_Type,
    "Learning_Disabilities": Learning_Disabilities,
    "Physical_Activity": Physical_Activity,
    "Family_Income": Family_Income,
    "Peer_Influence": Peer_Influence
}

# Predict and display result
if st.button("Predict Exam Score"):
    score = predict_exam_score(new_student)
    st.success(f"🎯 Predicted Exam Score: **{score:.2f}**")
