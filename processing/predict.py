
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def predict_model(
    X_test,
    test_df: pd.DataFrame,
    model_path: str = "models/random_forest.joblib",
) -> pd.DataFrame:
    """
    Predict unscheduled removal probabilities
    within the next 30 flight cycles.
    """

    # 1. Load trained model
    model = joblib.load(model_path)

    # 2. Predict probability of positive class
    positive_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X_test)[
        :, positive_index
    ]

    # 3. Build prediction results
    columns = [
        "record_id",
        "snapshot_date",
        "aircraft_id",
        "component_id",
        "flight_cycles",
        "unscheduled_removal_next_30_cycles",
    ]

    available_cols = [
        col for col in columns if col in test_df.columns
    ]

    results = test_df[available_cols].copy()
    results = results.reset_index(drop=True)

    results["risk_probability"] = probabilities

    return results



import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
)


def select_threshold(
    val_df: pd.DataFrame,
    probabilities,
    target_col="unscheduled_removal_next_30_cycles",
    min_recall=0.80,
    max_alert_rate=0.05,
):
    """
    Select threshold using validation data only.

    Objective:
    - Maximize recall
    - Alert rate <= 5% on every snapshot date
    """

    data = val_df.copy().reset_index(drop=True)

    scores = np.asarray(probabilities, dtype=float)

    if len(data) != len(scores):
        raise ValueError("Prediction length mismatch")

    if not np.isfinite(scores).all():
        raise ValueError("Invalid probabilities")

    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Probabilities must be between 0 and 1")

    if not data[target_col].isin([0, 1]).all():
        raise ValueError("Invalid target values")

    keys = [
        "snapshot_date",
        "aircraft_id",
        "component_id",
    ]

    if data.duplicated(keys).any():
        raise ValueError(
            "Multiple records for one component per day"
        )

    y_true = data[target_col].to_numpy()

    # All possible decision boundaries
    thresholds = np.r_[
        np.inf,
        np.sort(np.unique(scores))[::-1],
    ]

    results = []

    for threshold in thresholds:

        y_pred = (scores >= threshold).astype(int)

        # 1. Recall
        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        # 2. Precision
        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        # 3. Daily alert rate
        daily = data[["snapshot_date"]].copy()
        daily["alert"] = y_pred

        daily_rates = daily.groupby(
            "snapshot_date"
        )["alert"].mean()

        max_daily_alert_rate = daily_rates.max()

        alert_rate = y_pred.mean()

        if max_daily_alert_rate > max_alert_rate:
            continue

        results.append({
            "threshold": float(threshold),
            "recall": float(recall),
            "precision": float(precision),
            "alert_rate": float(alert_rate),
            "max_daily_alert_rate": float(
                max_daily_alert_rate
            ),
            "alerts": int(y_pred.sum()),
            "target_met": recall >= min_recall,
        })

    if not results:
        raise ValueError("No feasible threshold")

    # Maximize recall, then precision,
    # then minimize number of alerts
    results.sort(
        key=lambda x: (
            x["recall"],
            x["precision"],
            -x["alerts"],
        ),
        reverse=True,
    )

    best = results[0]

    print("Selected threshold:", best["threshold"])
    print("Recall:", best["recall"])
    print("Precision:", best["precision"])
    print("Alert rate:", best["alert_rate"])
    print("Target met:", best["target_met"])

    return best


import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


def find_best_threshold(
    y_true,
    probabilities,
    max_alert_rate=0.05,
):
    """
    Find threshold with maximum recall
    under the alert budget.
    """

    thresholds = np.r_[
        np.inf,
        np.sort(np.unique(probabilities))[::-1],
    ]

    best = None

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        alert_rate = predictions.mean()

        if alert_rate > max_alert_rate:
            continue

        recall = recall_score(
            y_true, predictions, zero_division=0
        )

        precision = precision_score(
            y_true, predictions, zero_division=0
        )

        candidate = {
            "threshold": float(threshold),
            "recall": float(recall),
            "precision": float(precision),
            "alert_rate": float(alert_rate),
        }

        if best is None or (
            recall,
            precision,
            -predictions.sum(),
        ) > (
            best["recall"],
            best["precision"],
            -best["alert_rate"] * len(y_true),
        ):
            best = candidate

    return best


def compare_models(
    X_val,
    y_val,
    max_alert_rate=0.05,
    min_recall=0.80,
):
    """
    Compare trained models using Validation Data.
    """

    model_paths = {
        "Logistic Regression":
            "models/logistic_regression.joblib",

        "Random Forest":
            "models/random_forest.joblib",
    }

    y_true = np.asarray(y_val)
    results = []

    for name, path in model_paths.items():

        # 1. Load trained model
        model = joblib.load(path)

        # 2. Predict probabilities
        positive_idx = list(model.classes_).index(1)

        probabilities = model.predict_proba(
            X_val
        )[:, positive_idx]

        # 3. Select threshold on Validation
        best = find_best_threshold(
            y_true,
            probabilities,
            max_alert_rate=max_alert_rate,
        )

        threshold = best["threshold"]

        # 4. Convert probabilities to predictions
        predictions = (
            probabilities >= threshold
        ).astype(int)

        # 5. Calculate metrics
        results.append({
            "Model": name,
            "PR_AUC": average_precision_score(
                y_true, probabilities
            ),
            "ROC_AUC": roc_auc_score(
                y_true, probabilities
            ),
            "Precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "Threshold": threshold,
            "Alerts_per_100": predictions.mean() * 100,
            "Target_Met": (
                best["recall"] >= min_recall
            ),
        })

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by=["Target_Met", "Recall", "PR_AUC"],
        ascending=[False, False, False],
    )

    return comparison
