from diffusers import AutoencoderKLWan, WanPipeline
from .base_model import BaseModel
import torch

class Wan13BModel(BaseModel):
    def load(self):
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        
        # VAE를 float16으로 변경하여 성능 개선
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        vae.enable_slicing()
        vae.enable_tiling()
        
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # 비디오 생성 특화 최적화
        pipe.enable_model_cpu_offload() # 주석처리 시 GPU 사용이 아닌 과도한 CPU 사용으로 속도 저하 매우 큼
        pipe.enable_attention_slicing("max") # 주석 처리 시 속도에 미치는 영향 없거나 적음

        return pipe
