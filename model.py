from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact,
    Metrics,
)


@component(
    packages_to_install=[
        "pandas",
        "gcsfs",
        "numpy==1.23.5",
        "scikit-learn==1.3.0",
        "joblib",
    ],
    output_component_file="training.yaml",
    base_image="python:3.10",
)
def training(df_train: Input[Dataset], trained_model: Output[Model]):

    import pandas as pd
    import os
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    df_train = pd.read_csv(df_train.path + ".csv")

    x = df_train.drop("Purchase", axis=1)
    y = df_train["Purchase"]

    regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

    regressor.fit(x, y)

    trained_model.metadata["framework"] = "RandomForestRegressor"
    os.makedirs(trained_model.path, exist_ok=True)
    joblib.dump(regressor, os.path.join(trained_model.path, "model.joblib"))


@component(
    packages_to_install=[
        "pandas",
        "numpy==1.23.5",
        "gcsfs",
        "scikit-learn==1.3.0",
        "joblib",
    ],
    output_component_file="model_evaluation.yaml",
    base_image="python:3.10",
)
def model_evaluation(
    test_set: Input[Dataset],
    training_model: Input[Model],
    kpi: Output[Metrics],
):
    import pandas as pd
    from math import sqrt
    import os
    import joblib
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    data = pd.read_csv(test_set.path + ".csv")
    file_name = os.path.join(training_model.path, "model.joblib")

    model = joblib.load(file_name)

    X_test = data.drop("Purchase", axis=1)
    y_test = np.array(data["Purchase"])

    xgb_y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, xgb_y_pred)
    mse = mean_squared_error(y_test, xgb_y_pred)
    r2 = r2_score(y_test, xgb_y_pred)
    rmse = sqrt(mean_squared_error(y_test, xgb_y_pred))

    training_model.metadata["mean_absolute_error"] = mae
    training_model.metadata["mean_squared_error"] = mse
    training_model.metadata["R2_Score"] = r2
    training_model.metadata["root_mean_absolute_error"] = rmse

    kpi.log_metric("mean_absolute_error", mae)
    kpi.log_metric("mean_squared_error", mse)
    kpi.log_metric("R2_Score", r2)
    kpi.log_metric("root_mean_absolute_error", rmse)


@component(
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
    base_image="python:3.10",
)
def deploy_xgboost_model(
    model: Input[Model],
    project_id: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    """Deploys an XGBoost model to Vertex AI Endpoint.

    Args:
        model: The model to deploy.
        project_id: The project ID of the Vertex AI Endpoint.

    Returns:
        vertex_endpoint: The deployed Vertex AI Endpoint.
        vertex_model: The deployed Vertex AI Model.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    deployed_model = aiplatform.Model.upload(
        display_name="bf_model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    )
    # endpoint = deployed_model.deploy(
    #     deployed_model_display='deployment_test'
    #     machine_type="n1-standard-4")

    # vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name
