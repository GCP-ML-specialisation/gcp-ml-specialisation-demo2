from kfp import compiler
import google.cloud.aiplatform as aip
from kfp import dsl
from pre_processing_modules import (
    basic_preprocessing,
    data_transformation,
    train_validation_test_split,
)

from model import (
    model_evaluation,
    training_hyperp_tuning,
    deploy_rf_model,
)

import configparser

config = configparser.ConfigParser()
config.read("config.ini")


PROJECT_ID = config["gcp_vars"]["PROJECT_ID"]
BUCKET_URI = config["gcp_vars"]["BUCKET_URI"]
SERVICE_ACCOUNT = config["gcp_vars"]["SERVICE_ACCOUNT"]
PIPELINE_ROOT = "{}/pipeline_root".format(BUCKET_URI)

aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI, service_account=SERVICE_ACCOUNT)


@dsl.pipeline(
    name="BF Pipeline",
    description="Pipeline to predict purchased amount using the Black Friday Dataset",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline(
    train_name: str = "train.csv",
    BUCKET_URI: str = BUCKET_URI,
    raw_folder: str = "raw_data/",
):

    pre_processed_df = basic_preprocessing(
        bucket_URI=BUCKET_URI,
        folder=raw_folder,
        train=train_name,
    )
    feature_engineered_df = data_transformation(
        df_train=pre_processed_df.outputs["dataset_train"],
    )
    ready_dataset = train_validation_test_split(
        df_train=feature_engineered_df.outputs["dataset_train"]
    )

    model = training_hyperp_tuning(
        df_train=ready_dataset.outputs["dataset_train"],
    )

    model_evaluation(
        test_set=ready_dataset.outputs["dataset_valid"],
        training_model=model.outputs["trained_model"],
    )

    deploy_rf_model(
        model=model.outputs["trained_model"],
        project_id=PROJECT_ID,
    )


compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=config["pipeline"]["template_path"],
)
