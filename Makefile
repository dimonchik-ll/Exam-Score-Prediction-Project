VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: venv
venv:
	python3 -m venv venv

.PHONY: install
install:
	$(PIP) install -r requirements.txt

models_and_pipelines/final_pipeline.pkl: src/model.py src/preprocessing_pipeline.py src/train.py
	$(PYTHON) src/train.py

.PHONY: backend frontend

backend:
	$(PYTHON) -m uvicorn src.api:app --reload

frontend:
	$(PYTHON) -m streamlit run src/app.py