from kfp.v2.dsl import component
import pandas as pd


@component(
    packages_to_install=["pandas", "gcsfs", "scikit-learn", "numpy"],
    output_component_file="pre_processing_data.yaml",
    base_image="python:3.9",
)
def pre_processed_bf_data(
    train_name: str,
    test_name: str,
    BUCKET_URI: str,
    raw_folder: str,
    pre_processed_folder: str,
) -> str:

    # Define the bucket URI
    df_train_path = BUCKET_URI + raw_folder + train_name
    df_test_path = BUCKET_URI + raw_folder + test_name

    df_train = pd.read_csv(df_train_path)
    df_test = pd.read_csv(df_test_path)

    from .pre_processing_modules import basic_preprocessing, feature_engineering

    df_train, df_test = basic_preprocessing(df_train, df_test)
    df_train, df_test = feature_engineering(df_train, df_test)

    # write_csv_GCS(
    #     df_train, df_test, BUCKET_URI, pre_processed_folder, train_name, test_name
    # )
