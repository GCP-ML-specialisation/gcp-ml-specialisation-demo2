import google.cloud.aiplatform as aip


PROJECT_ID = "pa-poc-mlspec-2"
BUCKET_URI = "gs://pa_poc_mlspec_2_pipeline/"
SERVICE_ACCOUNT = "121050757542-compute@developer.gserviceaccount.com"
PIPELINE_ROOT = "{}/pipeline_root".format(BUCKET_URI)
DISPLAY_NAME = "bf_pipeline_job_unique"


aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI, service_account=SERVICE_ACCOUNT)

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    enable_caching=True,
    template_path="bf_pipeline.json",
    pipeline_root=f"{BUCKET_URI}pipeline_root/",
)

job.run()
