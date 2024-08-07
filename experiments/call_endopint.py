import requests
import subprocess
import json

# Get the access token using gcloud
access_token = (
    subprocess.check_output("gcloud auth print-access-token", shell=True)
    .decode("utf-8")
    .strip()
)

ENDPOINT_ID = "3626204741568036864"
PROJECT_ID = "121050757542"
INPUT_DATA_FILE = "INPUT-JSON"

# Define the project ID, endpoint ID, and input data file
project_id = "121050757542"
endpoint_id = "3626204741568036864"

input_data = {"instances": [[1.0, 2.0, 17.0, 2.0, 0.0, 1.0, 5.0, 8.0]]}
# Define the endpoint URL
url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/endpoints/{endpoint_id}:predict"

# Define the headers
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
}

# Send the POST request
response = requests.post(url, headers=headers, json=input_data)

# Print the response
print(response.json())
