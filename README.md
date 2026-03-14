# Exam Score Prediction Project
## Overview

This project explores factors that influence students' exam performance and builds a machine learning model to predict exam scores based on various academic, behavioral, and environmental features.

## Dataset

The dataset contains synthetic student data with features related to study habits, learning environment, and lifestyle.

### Main Features

- **study_hours** — number of hours spent studying
- **class_attendance** — percentage of attended classes
- **sleep_hours** — average hours of sleep
- **sleep_quality** — subjective sleep quality score
- **gender** — student gender
- **age** — student age
- **course** — academic program
- **internet_access** — access to internet
- **facility_rating** — quality of learning facilities
- **study_method** — study method (coaching, self-study, etc.)
- **exam_difficulty** — perceived difficulty of the exam

### Target variable

- **exam_score** — final exam result

## Usage

- Clone the project to your local machine
```bash
git clone https://github.com/dimonchik-ll/Exam-Score-Prediction-Project.git
cd Exam-Score-Prediction-Project
```

- Create virtual environment
```bash
make venv
```

- Install dependencies
```bash
make install
```

- Running the backend
```bash
make backend
```

- Running the frontend
```bash
make frontend
```

The web app allows users to input student characteristics and receive a predicted exam score.