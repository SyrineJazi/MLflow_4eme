import sys
import argparse
import joblib
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model


def main():
    # Testing GIT third try
    # Testing the pipepline with a comment SECOND TRY
    # Identification des étapes via les arguments
    parser = argparse.ArgumentParser(
        description="Pipeline de Machine Learning")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Préparer les données")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Entraîner le modèle")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Évaluer le modèle")

    args = parser.parse_args()

    # Préparation des données
    if args.prepare:
        print("Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(
            "merged_data.csv", target_column="Churn")
        # Sauvegrade des données préparés
        joblib.dump((X_train, X_test, y_train, y_test), "prepared_data.joblib")
        print("Données préparées avec succès!")

    # Entraînement du modèle
    if args.train:
        print("Entraînement du modèle...")
        # Charger les données
        X_train, X_test, y_train, y_test = joblib.load("prepared_data.joblib")  
        model = train_model(X_train, y_train)
        save_model(model, "model.joblib")
        print("Modèle entraîné et sauvegardé avec succès!")

    # Évaluation du modèle
    if args.evaluate:
        print("Évaluation du modèle...")
        # Chargement du modèle sauvegardé et des données
        model = load_model("model.joblib")
        X_train, X_test, y_train, y_test = joblib.load("prepared_data.joblib")  
        accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)
        print(f"Précision du modèle: {accuracy:.2f}")
        print(f"Rapport de classification:\n{report}")


if __name__ == "__main__":
    main()
