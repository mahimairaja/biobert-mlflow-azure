import mlflow.pyfunc
from mlflow.models import set_model
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

class FastEmbedOnnx(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None
        TextEmbedding.add_custom_model(
          model="mahimairaja/biobert-onnx",
          pooling=PoolingType.MEAN,
          normalization=True,
          sources=ModelSource(hf="mahimairaja/biobert-onnx"),
          dim=768,
          model_file="model.onnx",
      )

    def load_context(self, context):
        model_file_path = context.artifacts["biobert_model"]
        self.model = TextEmbedding(model_name="mahimairaja/biobert-onnx", specific_model_path=model_file_path)

    def predict(self, context, model_input):
        if self.model is None:
            raise ValueError(
                "The model has not been loaded. "
                "Ensure that 'load_context' is properly executed."
            )
        return list(self.model.embed(model_input))

set_model(FastEmbedOnnx())