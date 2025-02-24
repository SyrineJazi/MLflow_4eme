import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Ensure models directory exists
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Data Preparation
def prepare_data(filepath, target_column, test_size=0.2, random_state=42):
    data = pd.read_csv(filepath)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Encode target variable if categorical
    label_encoder = None
    if y.dtype == "object":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test, scaler, label_encoder, label_encoders

# 2. Model Training
def train_model(X_train, y_train, random_state=42):
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

# 3. Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report
    

# 4. Save & Load Model
def save_model(model, filename="models/model.joblib"):
    joblib.dump(model, filename)

def load_model(filename="models/model.joblib"):
    return joblib.load(filename)

# 5. Save & Load Preprocessors
def save_preprocessor(obj, filename):
    joblib.dump(obj, os.path.join(MODELS_DIR, filename))

def load_preprocessor(filename):
    return joblib.load(os.path.join(MODELS_DIR, filename))

