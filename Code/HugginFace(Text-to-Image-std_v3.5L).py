#https://huggingface.co/stabilityai/stable-diffusion-3.5-large
#huggingface-cli login
#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

from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model_id = "stabilityai/stable-diffusion-3.5-large"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

prompt = ("a photo of a duck floating on the river")

image = pipeline(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=4.5,
    max_sequence_length=512,
).images[0]
image.save("a photo of an duck floating on the river.png")