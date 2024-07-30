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


PROJECT_ID = "pa-poc-mlspec-2"
BUCKET_URI = "gs://pa_poc_mlspec_2_pipeline/"
SERVICE_ACCOUNT = "121050757542-compute@developer.gserviceaccount.com"
PIPELINE_ROOT = "{}/pipeline_root".format(BUCKET_URI)

aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI, service_account=SERVICE_ACCOUNT)


@dsl.pipeline(
    name="BF Pipeline",
    description="Pipeline to predict purchased amount using the Black Friday Dataset",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline(
    train_name: str = "train.csv",
    test_name: str = "test.csv",
    BUCKET_URI: str = BUCKET_URI,
    raw_folder: str = "raw_data/",
):

    pre_processed_df = basic_preprocessing(
        bucket_URI=BUCKET_URI,
        folder=raw_folder,
        train=train_name,
        test=test_name,
    )
    feature_engineered_df = data_transformation(
        df_train=pre_processed_df.outputs["dataset_train"],
        df_test=pre_processed_df.outputs["dataset_test"],
    )
    ready_dataset = train_validation_test_split(
        df_train=feature_engineered_df.outputs["dataset_train"]
    )

    model = training_hyperp_tuning(df_train=ready_dataset.outputs["dataset_train"])

    model_evaluation(
        test_set=ready_dataset.outputs["dataset_valid"],
        training_model=model.outputs["trained_model"],
    )

    deploy_rf_model(
        model=model.outputs["trained_model"],
        project_id=PROJECT_ID,
    )


compiler.Compiler().compile(pipeline_func=pipeline, package_path="bf_pipeline.json")
