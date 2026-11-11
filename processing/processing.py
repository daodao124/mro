
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

TARGET = "unscheduled_removal_next_30_cycles"

NUMERIC_FEATURES = [
    "flight_cycles", "cycles_since_install", "vibration_rms",
    "temperature_c", "pressure_delta", "fault_count_30d",
    "maintenance_count_90d", "last_maintenance_cycles_ago",
    "has_prior_maintenance", "has_recent_fault", "has_recent_maintenance",
    "log_cycles_since_install", "vibration_temperature_interaction",
]
CATEGORICAL_FEATURES = [
    "aircraft_model", "component_type", "component_position", "fault_code",
    "snapshot_month", "snapshot_day", "snapshot_weekday",
]


def clean_mro_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based cleanup only; do not learn medians or quantiles here."""
    data = df.copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce", utc=True)
    if data["snapshot_date"].isna().any():
        raise ValueError("Invalid snapshot_date")

    keys = ["snapshot_date", "aircraft_id", "component_position"]
    if data[keys].isna().any().any():
        raise ValueError("Missing observation keys")
    duplicates = data.loc[data.duplicated(keys, keep=False)]
    if not duplicates.empty:
        other_cols = [c for c in data.columns if c not in [*keys, "record_id"]]
        different = (duplicates.groupby(keys, dropna=False)[other_cols].nunique(dropna=False) > 1)
        if different.any().any():
            raise ValueError("Same aircraft/position/date has conflicting records")
        data = data.drop_duplicates(keys, keep="first")

    nonnegative = [
        "flight_cycles", "cycles_since_install", "vibration_rms",
        "fault_count_30d", "maintenance_count_90d", "last_maintenance_cycles_ago",
    ]
    for col in nonnegative:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        # Negative values impossible for these fields; mark invalid as missing.
        data.loc[data[col] < 0, col] = np.nan

    for col in ["temperature_c", "pressure_delta"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.reset_index(drop=True)


def report_missing_outliers(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Descriptive audit only. Nothing learned here is used for model transform."""
    missing = df.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False)
    rows = []
    for col in ["vibration_rms", "temperature_c", "pressure_delta"]:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        rows.append({"feature": col, "outlier_count": int(mask.sum()),
                     "outlier_rate": float(mask.mean())})
    return missing, pd.DataFrame(rows)


def engineer_mro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features only. No future events or target in calculations."""
    data = df.copy()
    dt = data["snapshot_date"].dt

    # Calendar fields are already known at snapshot time.
    data["snapshot_month"] = dt.month.astype("string")
    data["snapshot_day"] = dt.day.astype("string")
    data["snapshot_weekday"] = dt.dayofweek.astype("string")  # Monday = 0

    # Usage and recent-history fields, all from the current snapshot.
    data["has_prior_maintenance"] = data["last_maintenance_cycles_ago"].notna().astype(int)
    data["has_recent_fault"] = (data["fault_count_30d"] > 0).astype(int)
    data["has_recent_maintenance"] = (data["maintenance_count_90d"] > 0).astype(int)
    data["log_cycles_since_install"] = np.log1p(data["cycles_since_install"])
    data["vibration_temperature_interaction"] = (
        data["vibration_rms"] * data["temperature_c"]
    )
    return data


def report_imbalance(df: pd.DataFrame, target_col: str = TARGET) -> None:
    """Analyze target distribution before split; do not resample here."""
    counts = df[target_col].value_counts().sort_index()
    print("Target counts:", counts.to_dict())
    print("Positive rate:", round(100 * df[target_col].eq(1).mean(), 2), "%")


def split_mro_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    label_col: str = "label_available_at",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological cutoffs by row quantiles, never split a snapshot date."""
    if not (0 < train_ratio < train_ratio + val_ratio < 1):
        raise ValueError("Invalid train/validation ratios")

    data = df.copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], utc=True, errors="coerce")
    if label_col not in data:
        raise ValueError("label_available_at is required to verify the 30-cycle target horizon")
    data[label_col] = pd.to_datetime(data[label_col], utc=True, errors="coerce")
    if data[["snapshot_date", label_col]].isna().any().any():
        raise ValueError("Missing snapshot date or label availability")
    if (data[label_col] < data["snapshot_date"]).any():
        raise ValueError("Label available before snapshot")

    data = data.sort_values(["snapshot_date", "aircraft_id", "component_id"]).reset_index(drop=True)
    n = len(data)
    if n < 3:
        raise ValueError("Not enough records")
    val_start = data.iloc[int(n * train_ratio)]["snapshot_date"]
    test_start = data.iloc[int(n * (train_ratio + val_ratio))]["snapshot_date"]
    if val_start >= test_start:
        raise ValueError("Not enough distinct dates for train/validation/test")

    train_period = data["snapshot_date"] < val_start
    val_period = data["snapshot_date"].between(val_start, test_start, inclusive="left")
    test_period = data["snapshot_date"] >= test_start

    # A training label must have been known by the next period's start.
    train = data[train_period & (data[label_col] < val_start)].copy()
    val = data[val_period & (data[label_col] < test_start)].copy()
    test = data[test_period].copy()

    print("Purged train rows:", int(train_period.sum()) - len(train))
    print("Purged validation rows:", int(val_period.sum()) - len(val))
    for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
        if part.empty or part[TARGET].nunique() < 2:
            raise ValueError(f"{name} empty or missing a class")
        print(f"{name}: {len(part):,} | {part['snapshot_date'].min().date()}"
              f" → {part['snapshot_date'].max().date()} | positive {part[TARGET].mean():.2%}")

    for col in ["aircraft_id", "component_id", "installation_id"]:
        if col in data:
            overlaps = len(set(train[col]) & set(test[col]))
            print(f"Train/Test overlap {col}: {overlaps} (audit; not always leakage)")
    return train, val, test


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Optional sensor clipping; bounds always learned from training rows."""
    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not (0 <= self.lower < self.upper <= 1):
            raise ValueError("Invalid quantile range")
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.quantile(arr, self.lower, axis=0)
        self.upper_ = np.quantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


def fit_preprocessing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clip_outliers: bool = False,
) -> dict:
    """Fit all learned transformations and balancing parameters on training only."""
    sensor_features = ["vibration_rms", "temperature_c", "pressure_delta"]
    other_numeric = [col for col in NUMERIC_FEATURES if col not in sensor_features]

    sensor_steps = [("imputer", SimpleImputer(strategy="median"))]
    if clip_outliers:
        sensor_steps.append(("clipper", QuantileClipper()))
    sensor_steps.append(("scaler", RobustScaler()))

    preprocessor = ColumnTransformer([
        ("sensors", Pipeline(sensor_steps), sensor_features),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]), other_numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])

    def get_x(part):
        X = part[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("string").fillna("UNKNOWN").astype(object)
        return X

    X_train = preprocessor.fit_transform(get_x(train_df))
    X_val = preprocessor.transform(get_x(val_df))
    X_test = preprocessor.transform(get_x(test_df))

    y_train = train_df[TARGET].astype(int).copy()
    y_val = val_df[TARGET].astype(int).copy()
    y_test = test_df[TARGET].astype(int).copy()
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.to_numpy())
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.get_feature_names_out(),
        "class_weights": class_weights,
        "scale_pos_weight": int((y_train == 0).sum()) / int((y_train == 1).sum()),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_mro_10000.csv")
    df = clean_mro_data(df)
    missing, outliers = report_missing_outliers(df)
    print("Missing:\n", missing)
    print("Outliers (audit only):\n", outliers)

    df = engineer_mro_features(df)
    report_imbalance(df)
    train_df, val_df, test_df = split_mro_data(df)
    result = fit_preprocessing(train_df, val_df, test_df)
    for key in ("X_train", "X_val", "X_test"):
        print(key, result[key].shape)
    print("Class weights:", result["class_weights"])
