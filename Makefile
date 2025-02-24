SHELL := /bin/bash

# Variables
PYTHON = python3
VENV = venv
REQ = requirements.txt
MAIN = main.py
MODEL_FILE = model.joblib

# Installer dépendances et créer Virtual Env si n'existent pas
install:
	if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV) && source $(VENV)/bin/activate && pip install -r $(REQ); \
	else \
		echo "L'environnement virtuel existe déjà, activation..."; \
		source $(VENV)/bin/activate && pip install -r $(REQ); \
	fi

# Vérification du code (formatage et qualité)
check:
	flake8 --max-line-length=100 model_pipeline.py main.py

# Corriger le code
correct:
	 autopep8 --in-place --aggressive --aggressive model_pipeline.py main.py

# Étape de protection du code (sécurisation)
# protect:
#	bandit -r . --quiet  
#	safety check --full-report  
#	detect-secrets --scan 

# Préparation des données
prepare:
	$(PYTHON) $(MAIN) --prepare

# Entraînement du modèle
train:
	$(PYTHON) $(MAIN) --train

# Évaluation du modèle
evaluate:
	$(PYTHON) $(MAIN) --evaluate

# Exécution des tests
test:
	pytest tests/

# Nettoyer les fichiers temporaires
clean:
	rm -rf _pycache_ *.pkl *.joblib
