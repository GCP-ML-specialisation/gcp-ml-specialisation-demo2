import google.cloud.aiplatform as aip


PROJECT_ID = "prj-dev-mlbf-flt-01-29e5"
BUCKET_URI = "gs://pipeline_black_friday/"
SERVICE_ACCOUNT = (
    "project-service-account@prj-dev-mlbf-flt-01-29e5.iam.gserviceaccount.com"
)
PIPELINE_ROOT = "{}/pipeline_root".format(BUCKET_URI)
DISPLAY_NAME = "bf_pipeline_job_unique"


aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI, service_account=SERVICE_ACCOUNT)

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    enable_caching=False,
    template_path="bf_pipeline.json",
    pipeline_root=f"{BUCKET_URI}pipeline-root/",
)

job.run()
