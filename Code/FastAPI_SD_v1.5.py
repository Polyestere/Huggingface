# FastApi_sdv1.5(Text-to-Image)
# http://localhost:8000/docs (사이트)
# uvicorn FastAPI_SD_v1.5:app
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from diffusers import StableDiffusionPipeline
import torch
import os

app = FastAPI(title="Stable Diffusion 1.5 API", description="텍스트를 입력받아 이미지를 생성하는 API입니다.", version="1.0")

# 모델 로딩 (최초 1회)
pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 출력 폴더 생성
os.makedirs("outputs", exist_ok=True)

@app.get("/")
def root():
    return {"message": "FastAPI is running. Try /docs for Swagger UI."}

@app.post("/generate", summary="이미지 생성", description="텍스트 프롬프트를 입력받아 이미지를 생성합니다.")
def generate_image(prompt: str = Form(...)):
    """
    텍스트 프롬프트를 기반으로 이미지를 생성하여 반환합니다.
    """
    image = pipe(prompt=prompt, guidance_scale=7.5).images[0]
    output_path = "outputs/result.png"
    image.save(output_path)
    return FileResponse(output_path, media_type="image/png")
