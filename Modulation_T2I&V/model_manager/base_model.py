### model_manager/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def load(self):
        pass
