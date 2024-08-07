from kfp.v2.dsl import component, Input, Output, Dataset


@component(
    packages_to_install=[
        "pandas",
        "gcsfs",
        "scikit-learn==1.3.0",
        "numpy==1.23.5",
    ],
    output_component_file="./pipeline_components_file/feature_engineering.yaml",
    base_image="python:3.10",
)
def data_transformation(
    df_train: Input[Dataset],
    dataset_train: Output[Dataset],
):

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    df_train = pd.read_csv(df_train.path + ".csv")

    # Handle categorical to integer transformation for 'Gender'
    gender_mapping = {"F": 0, "M": 1}
    df_train["Gender"] = df_train["Gender"].map(gender_mapping)

    # Columns to encode
    cols = ["Age", "City_Category", "Stay_In_Current_City_Years"]

    combined_df = df_train[cols]

    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Apply LabelEncoder to each column and transform back to DataFrame
    for col in cols:
        combined_df[col] = le.fit_transform(combined_df[col])

    # Split the combined data back into train and test sets
    df_train[cols] = combined_df
    df_train["Purchase"] = np.log1p(df_train["Purchase"])
    df_train.to_csv(dataset_train.path + ".csv", index=False)


@component(
    packages_to_install=[
        "pandas",
        "gcsfs",
        "scikit-learn==1.3.0",
        "numpy==1.23.5",
    ],
    output_component_file="./pipeline_components_file/basic_preprocessing.yaml",
    base_image="python:3.10",
)
def basic_preprocessing(
    bucket_URI: str,
    folder: str,
    train: str,
    dataset_train: Output[Dataset],
):

    import pandas as pd

    df_train_uri = "".join([bucket_URI, folder, train])

    df_train = pd.read_csv(df_train_uri)

    df_train["Stay_In_Current_City_Years"] = df_train[
        "Stay_In_Current_City_Years"
    ].str.replace("+", "")
    df_train["Stay_In_Current_City_Years"] = df_train[
        "Stay_In_Current_City_Years"
    ].astype(int)

    ## Dropping User_id and Product_ID
    df_train = df_train.drop("User_ID", axis=1)
    df_train = df_train.drop("Product_ID", axis=1)
    df_train = df_train.drop("Product_Category_3", axis=1)

    ## Imputing missing values with mode
    df_train["Product_Category_2"].mode()[0]
    df_train["Product_Category_2"] = df_train["Product_Category_2"].fillna(
        df_train["Product_Category_2"].mode()[0]
    )

    df_train.to_csv(dataset_train.path + ".csv", index=False)


@component(
    packages_to_install=[
        "pandas",
        "gcsfs",
        "numpy==1.23.5",
        "scikit-learn==1.3.0",
    ],
    output_component_file="./pipeline_components_file/train_validation_test_split.yaml",
    base_image="python:3.10",
)
def train_validation_test_split(
    df_train: Input[Dataset],
    dataset_train: Output[Dataset],
    dataset_valid: Output[Dataset],
    validation_size: float = 0.2,
):

    import pandas as pd
    from sklearn.model_selection import train_test_split

    df_train = pd.read_csv(df_train.path + ".csv")

    df_train, df_valid = train_test_split(
        df_train, test_size=validation_size, random_state=42
    )

    df_train.to_csv(dataset_train.path + ".csv", index=False)
    df_valid.to_csv(dataset_valid.path + ".csv", index=False)
