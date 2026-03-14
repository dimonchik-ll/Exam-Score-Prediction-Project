import streamlit as st
from src.data import data
import requests

st.title("Exam score prediction")

with st.form("Input student data"):
    age = st.number_input("Age", min_value=0, value=None, step=1, placeholder="Enter your age")
    gender = st.selectbox("Gender", data["gender"].unique(), index=None, placeholder="Select your gender")
    course = st.selectbox("Course", data["course"].unique(), index=None, placeholder="Select your course")
    study_hours = st.number_input("Study hours", min_value=0.0, value=None, step=0.1, placeholder="Enter how many hours you study per day")
    class_attendance = st.number_input("Class attendance", min_value=0.0, value=None, step=0.1, placeholder="Enter the percentage of classes you attended")
    internet_access = st.selectbox("Internet access", data["internet_access"].unique(), index=None, placeholder="Do you have access to the internet")
    sleep_hours = st.number_input("Sleeping time", min_value=0.0, value=None, step=0.1, placeholder="Enter how many hours you sleep per day")
    sleep_quality = st.selectbox("Sleep quality", data["sleep_quality"].unique(), index=None, placeholder="Estimate quality of your sleep")
    study_method = st.selectbox("Study method", data["study_method"].unique(), index=None, placeholder="Select your study method")
    facility_rating = st.selectbox("Facility rating", data["facility_rating"].unique(), index=None, placeholder="Estimate your facility rating")
    exam_difficulty = st.selectbox("Exam difficulty", data["exam_difficulty"].unique(), index=None, placeholder="Select difficulty of the exam")
    submit = st.form_submit_button("Apply")

if submit:

    student_data = {
        "age": age,
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty
        }
    
    response = requests.post("http://127.0.0.1:8000/score", json=student_data)
    st.success(f"Predicted Exam Score: {response.json()["score"]:.2f}")