from diffusers import StableDiffusion3Pipeline
from .base_model import BaseModel
import torch

class SD35MModel(BaseModel):
    def load(self):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        # CPU 오프로딩으로 메모리 절약
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        
        # VAE 최적화
        if hasattr(pipe, 'vae'):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        
        return pipe
