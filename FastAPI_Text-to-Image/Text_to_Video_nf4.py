# uvicorn Text_to_Video:app
# https://huggingface.co/facebook/nllb-200-distilled-600M
# https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
# http://127.0.0.1:8000/generate-video
# pip install --upgrade diffusers bitsandbytes


import os
import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel
from diffusers import WanPipeline, AutoencoderKLWan, BitsAndBytesConfig
from diffusers.utils import export_to_video

app = FastAPI()

# NF4 양자화 설정
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로딩 (NF4 양자화 적용)
model_id = "sarthak247/Wan2.1-T2V-1.3B-nf4"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(
    model_id,
    vae=vae,
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

os.makedirs("outputs(video)", exist_ok=True)

def translate(prompt):
    model_name = "facebook/nllb-200-distilled-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.src_lang = "kor_Hang"
    inputs = tokenizer(prompt, return_tensors="pt")
    translated = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn")
    )
    translated_prompt = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(translated_prompt)
    return translated_prompt

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate-video")
def generate_video(request: PromptRequest):
    prompt = translate(request.prompt)
    video = pipe(
        prompt=prompt,
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        height=480,
        width=640,
        num_inference_steps=5,
        num_frames=17,
        guidance_scale=5
    ).frames[0]
    export_to_video(video_frames="outputs(video)/output.mp4", fps=8)
    output_path = "outputs(video)/output.mp4"
    video.save(output_path)
    return {"message": "OK", "file_path": output_path}