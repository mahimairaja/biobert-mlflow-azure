import time
import mlflow
import mlflow.sklearn
import pandas as pd
from azureml.core import Workspace

auth = {
    "subscription_id": "84d2e25f-0327-4359-a986-106d736282b9",
    "resource_group": "rg-govt-assist-dev-eastus2",
    "workspace_name": "aml-govt-assist-dev"
}
ws = Workspace(**auth)
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

model_uri = "models:/medical_relevance_classifier_dummy/1"
model = mlflow.sklearn.load_model(model_uri)

df = pd.DataFrame([
    [0.0] * 2
])

start_time = time.time()
result = model.predict(df)
result_proba = model.predict_proba(df)

print(result)
print(result_proba)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
