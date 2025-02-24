import os
import mlflow
import mlflow.sklearn
import argparse
from model_pipeline import prepare_data, train_model, evaluate_model,save_model, load_model, save_preprocessor, load_preprocessor

# Ensure models directory exists
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline with MLflow")
    parser.add_argument("--train", action="store_true", help="Train and save model with MLflow")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--save", type=str, default="models/model.joblib", help="Model save path")

    args = parser.parse_args()

    # Set MLflow experiment
    mlflow.set_experiment("syrine_jazi_DS9_ml_project")

    # Prepare data
    print(f"Loading dataset: {args.data}")
    X_train, X_test, y_train, y_test, scaler, label_encoder, label_encoders = prepare_data(args.data, args.target)

    if args.train:
        with mlflow.start_run():
            print("Training model...")
            model = train_model(X_train, y_train)

            print("Evaluating model...")
            accuracy, report = evaluate_model(model, X_test, y_test)

            # Log hyperparameters & metrics
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", accuracy)

            # Log model
            mlflow.sklearn.log_model(model, "random_forest_model")

            # Save model and preprocessors
            save_model(model, args.save)
            save_preprocessor(scaler, "scaler.pkl")
            save_preprocessor(label_encoder, "label_encoder.pkl")
            save_preprocessor(label_encoders, "label_encoders.pkl")

            print(f"Model trained and saved at: {args.save} (Accuracy: {accuracy:.2f})")

    if args.evaluate:
        if not os.path.exists(args.save):
            print(f"ERROR: {args.save} not found. Run `make train` first.")
            return

        print(f"Loading model from {args.save}...")
        model = load_model(args.save)
        accuracy, report = evaluate_model(model, X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        print("/nClassification report : ", report)

if __name__ == "__main__":
    main()
