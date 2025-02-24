SHELL := /bin/bash

# Variables
PYTHON = python3
VENV = ../venv
REQ = requirements.txt
MAIN = main.py
MODEL_FILE = models/model.joblib
DATA = merged_data.csv

# Install dependencies
install:
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV) && $(VENV)/bin/pip install -r $(REQ); \
	else \
		echo "Virtual environment exists, updating dependencies..."; \
		$(VENV)/bin/pip install -r $(REQ); \
	fi

# Start MLflow UI
mlflow-ui:
	$(VENV)/bin/mlflow ui --host 0.0.0.0 --port 5000 &

# Start MLflow with SQLite
mlflow-db:
	$(VENV)/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &

# Train model
train:
	$(VENV)/bin/python $(MAIN) --data $(DATA) --target Churn --train --save $(MODEL_FILE)

# Evaluate model
evaluate:
	$(VENV)/bin/python $(MAIN) --data $(DATA) --target Churn --evaluate --save $(MODEL_FILE)

# Clean up
clean:
	rm -rf __pycache__ models/*.pkl models/*.joblib $(VENV)
