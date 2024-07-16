import pandas as pd
import gcsfs


def read_csv_GCS(df_train_path: str, df_test_path: str):

    # Read the CSV file directly using pandas
    df_train = pd.read_csv(df_train_path)
    df_test = pd.read_csv(df_test_path)

    return df_train, df_test


def write_csv_GCS(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    bucket_uri: str,
    folder: str,
    df_train_filename: str,
    df_test_filename: str,
):
    full_path_train = f"{bucket_uri}{folder}{df_train_filename}"
    full_path_test = f"{bucket_uri}{folder}{df_test_filename}"

    # Create a GCS filesystem object
    fs = gcsfs.GCSFileSystem()

    with fs.open(full_path_train, "w") as f:
        df_train.to_csv(f, index=False)

    with fs.open(full_path_test, "w") as f:
        df_test.to_csv(f, index=False)
