import google.cloud.aiplatform as aip
import datetime
import configparser
import datetime

config = configparser.ConfigParser()
config.read("config.ini")

ct = str(datetime.datetime.now().timestamp())


PROJECT_ID = config["gcp_vars"]["PROJECT_ID"]
BUCKET_URI = config["gcp_vars"]["BUCKET_URI"]
SERVICE_ACCOUNT = config["gcp_vars"]["SERVICE_ACCOUNT"]
PIPELINE_ROOT = "{}/{}".format(BUCKET_URI, config["gcp_vars"]["PIPELINE_ROOT"])
DISPLAY_NAME = "{}{}".format(config["gcp_vars"]["PIPELINE_NAME"], ct)


aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI, service_account=SERVICE_ACCOUNT)

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    enable_caching=False,
    template_path=config["pipeline"]["template_path"],
    pipeline_root=f"{BUCKET_URI}pipeline_root/",
)

job.run()
