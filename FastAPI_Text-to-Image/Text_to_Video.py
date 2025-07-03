# uvicorn Text_to_Video:app
# https://huggingface.co/facebook/nllb-200-distilled-600M
# https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
# http://127.0.0.1:8000/generate-video
<<<<<<< HEAD

=======
>>>>>>> bfb7c82ec44217c5614ca7339a4ee0a188cecdfd

import os
import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

app = FastAPI()

# 모델 로딩 
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# 출력 폴더 생성
os.makedirs("outputs(video)", exist_ok=True)

#번역 모델(한국어 프롬프트 영어 번역)
def translate(prompt):
    model_name = "facebook/nllb-200-distilled-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # NLLB 모델의 경우 소스 언어와 타겟 언어 토큰을 추가해야 함
    tokenizer.src_lang = "kor_Hang"  # 한국어
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 영어로 번역 (eng_Latn)
    translated = model.generate(
    **inputs, 
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn")
    )
    translated_prompt = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(translated_prompt)
    return translated_prompt
class PromptRequest(BaseModel):
    prompt: str

#Post(번역 함수 호출 영상 생성 후 메세지 및 경로 반환)
#Body가 Json 형태
@app.post("/generate-video")
def generate_video(request: PromptRequest):
    prompt = translate(request.prompt)
    video = pipe(prompt=prompt,     
                 negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                 height=480,
                 width=640, 
                 num_inference_steps = 5, 
                 num_frames = 17,
                 guidance_scale=5
                ).frames[0]
    export_to_video( video_frames= "outputs(video)/output.mp4", fps=8)
    output_path = "outputs(video)/output.mp4"
    video.save(output_path)
    return {"message":"OK", "file_path": output_path}


