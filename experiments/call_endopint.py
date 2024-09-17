import requests
import subprocess
from pprint import pprint

# Get the access token using gcloud
access_token = (
    subprocess.check_output("gcloud auth print-access-token", shell=True)
    .decode("utf-8")
    .strip()
)


# Define the project ID, endpoint ID, and input data file
project_id = "121050757542"
endpoint_id = "1928034321235443712"

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
pprint(response.json())
