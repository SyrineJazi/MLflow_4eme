import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Preparation des donnees


def prepare_data(filepath, target_column, test_size=0.2, random_state=42):
    data = pd.read_csv(filepath)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Encode the target variable if it is categorical
    if y.dtypes == "object":
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Scale the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test

# 2. Entrainement du modele


def train_model(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

# 3. Evaluation du modele


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, conf_matrix
# 4. Sauvegarde du modele


def save_model(model, filepath):
    joblib.dump(model, filepath)

# 5. Chargement du modele


def load_model(filepath):
    return joblib.load(filepath)

# 6. Affichage de la matrice de confusion


def plot_confusion_matrix(conf_matrix, labels, title="Matrice de Confusion"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Predictions", fontsize=14)
    plt.ylabel("Valeurs Reelles", fontsize=14)
    plt.show()
