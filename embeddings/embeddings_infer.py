import mlflow
import pandas as pd
from azureml.core import Workspace
import time

ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# model_uri = "runs:/af05ef02-229a-4f44-bdc4-e17516087880/model"
model_uri = "models:/medical_relevance_embeddings_model/1"
model = mlflow.pyfunc.load_model(model_uri)

df = pd.DataFrame(["This is a test sentence."], columns=["text"])

start_time = time.time()
embeddings = model.predict(df)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

print("✅ Inference successful.")
print("Embedding preview (first 5 dims):", embeddings[0][:5])
print(type(embeddings))
print(type(embeddings[0]))
