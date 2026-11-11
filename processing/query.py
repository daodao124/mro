from pathlib import Path
import pandas as pd


from processing.contanst import BASE_DIR, DATA_PATH


def load_data ():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot_date"],
    )

    return df


