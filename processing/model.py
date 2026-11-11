
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_models(
    X_train,
    y_train,
    class_weights,
    preprocessor,
    output_dir="models",
):
    """
    Train and save Logistic Regression,
    Random Forest and fitted preprocessor.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(
            class_weight=class_weights,
            max_iter=2000,
            random_state=42,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        # Save trained model
        model_path = output_path / f"{name}.joblib"
        joblib.dump(model, model_path)

        trained_models[name] = model

        print(f"Saved model: {model_path}")

    # Save fitted preprocessor
    preprocessor_path = output_path / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Saved preprocessor: {preprocessor_path}")

    return trained_models
