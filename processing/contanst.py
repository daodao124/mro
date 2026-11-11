from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "synthetic_mro_10000.csv"

TARGET = "unscheduled_removal_next_30_cycles"

SENSOR_COLS = [
    "vibration_rms",
    "temperature_c",
    "pressure_delta",
]

NUMERIC_COLS = [
    "flight_cycles",
    "cycles_since_install",
    "fault_count_30d",
    "maintenance_count_90d",
    "last_maintenance_cycles_ago",
]

CATEGORICAL_COLS = [
    "aircraft_model",
    "component_type",
    "component_position",
    "fault_code",
]