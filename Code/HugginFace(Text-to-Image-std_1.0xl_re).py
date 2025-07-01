#https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
#프롬프트만으로 이미지 생성 X, 기존 이미지를 바탕으로 프롬프트에 맞춰 재구성
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

init_image = load_image(url).convert("RGB")
prompt = "a photo of an duck floating on the river"
image = pipe(prompt, image=init_image).images
image.save("a photo of an duck floating on the river_std_1.0xl_re.png")
