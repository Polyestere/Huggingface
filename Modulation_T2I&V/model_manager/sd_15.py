### model_manager/sd_15.py
from diffusers import StableDiffusionPipeline
from .base_model import BaseModel
import torch

class SD15Model(BaseModel):
    def load(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            variant="fp16"
        )
        
        pipe.enable_attention_slicing()

        return pipe.to("cuda") if torch.cuda.is_available() else pipe