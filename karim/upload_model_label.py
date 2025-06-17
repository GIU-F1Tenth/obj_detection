from roboflow import Roboflow

rf = Roboflow(api_key="20WGTaIu61u17gE3dQLn")
workspace = rf.workspace("giuf1tenth")

workspace.deploy_model(
    model_type="yolov11",                # or yolov5/yolov8 if needed
    model_path="IC_colab_model",         # this is the folder
    project_ids=["ic_2"],
    model_name="best"
)
