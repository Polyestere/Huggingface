#https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
from diffusers import StableDiffusionPipeline
import torch

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an duck floating on the river"
image = pipe(prompt).images[0]  
    
image.save("duck_floating_on_the_river_std_v1.5.png")

