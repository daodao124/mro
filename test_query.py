
from processing.query import load_data


def test_load_data():
    df = load_data()

    print("\nDataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    assert not df.empty, "Dataset is empty"
    assert "snapshot_date" in df.columns
    assert df["snapshot_date"].dtype.kind == "M"

    print("\nLoad data test passed!")


if __name__ == "__main__":
    test_load_data()
