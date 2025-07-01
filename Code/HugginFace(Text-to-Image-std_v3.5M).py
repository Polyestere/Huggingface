#https://huggingface.co/stabilityai/stable-diffusion-3.5-medium

import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "a photo of an duck floating on the river",
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]
image.save("a photo of an duck floating on the river_std_v3.5M.png")
