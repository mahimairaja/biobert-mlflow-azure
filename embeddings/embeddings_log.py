import mlflow
from azureml.core import Workspace

ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model='model.py',
        artifacts={"biobert_model": "model/biobert_model"},
        conda_env="model/conda.yaml",
        signature=mlflow.models.infer_signature(
            ["Hello world"], [[0.0]*768]
        ),
        registered_model_name="medical_relevance_embeddings_model"
    )

    print('\n\n\n\n')
    print(repr(model_info))
    print(repr(model_info.model_uri))
    print('\n\n\n\n')