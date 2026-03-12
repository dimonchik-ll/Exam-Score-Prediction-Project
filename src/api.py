import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from fastapi import FastAPI
from pydantic import BaseModel

class StudentData(BaseModel):
    age: int
    gender: str
    course: str
    study_hours: float
    class_attendance: float
    internet_access: str	
    sleep_hours: float
    sleep_quality: str
    study_method: str
    facility_rating: str
    exam_difficulty: str

app = FastAPI()

model = joblib.load("models_and_pipelines/final_pipeline.pkl")

@app.post("/score")
def score(data: StudentData):
    X = pd.DataFrame([{
        "age": data.age,
        "gender": data.gender,
        "course": data.course,
        "study_hours": data.study_hours,
        "class_attendance": data.class_attendance,
        "internet_access": data.internet_access,
        "sleep_hours": data.sleep_hours,
        "sleep_quality": data.sleep_quality,
        "study_method": data.study_method,
        "facility_rating": data.facility_rating,
        "exam_difficulty": data.exam_difficulty
    }])

    prediction = model.predict(X)[0]

    return {"score": float(prediction)}