import os
import roboflow

PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")
WORKSPACE_ID = os.getenv("WORKSPACE_ID")

rf = roboflow.Roboflow(api_key=API_KEY)
workspace = rf.workspace(WORKSPACE_ID)

workspace.upload_dataset(
    dataset_path="./ds/v2-110",
    project_name=PROJECT_ID,
    project_type="semantic-segmentation",
    is_prediction=True
)

print("Upload complete!")