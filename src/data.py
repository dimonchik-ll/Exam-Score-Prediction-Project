import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("datasets/Exam_Score_Prediction.csv")

X = data.drop("exam_score", axis=1)
y = data["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=42)