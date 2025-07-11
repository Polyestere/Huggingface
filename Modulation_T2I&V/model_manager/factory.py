from .sd_15 import SD15Model
from .sd_35m import SD35MModel
from .wan_13b import Wan13BModel

MODEL_CLASS_MAP = {
    "SD_1.5": SD15Model,
    "SD_3.5M": SD35MModel,
    "Wan_1.3B": Wan13BModel,
}

def get_model_instance(model_name):
    model_class = MODEL_CLASS_MAP.get(model_name)
    if model_class is None:
        raise ValueError(f"Unsupported model: {model_name}")
    return model_class(model_name)
