import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from azureml.core import Workspace

ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
experiment_name = "medical-classification-dummy"
mlflow.set_experiment(experiment_name)

data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 
    'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[['feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("random_state", 1)
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="medical_relevance_classifier_dummy"
    )