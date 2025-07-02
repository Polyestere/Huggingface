# uvicorn Text_to_Image:app
# https://huggingface.co/facebook/nllb-200-distilled-600M
# https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
# http://127.0.0.1:8000/generate-image

from fastapi import FastAPI
from diffusers import StableDiffusion3Pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel
import torch
import os

app = FastAPI()

# 모델 로딩 
pipe = StableDiffusion3Pipeline.from_pretrained(
   "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

# 출력 폴더 생성
os.makedirs("outputs(image)", exist_ok=True)

#번역 모델(한국어 프롬프트 영어 번역)
def translate(prompt):
    model_name = "Helsinki-NLP/opus-mt-tc-big-ko-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    translated = model.generate(**inputs)
    translated_prompt = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(translated_prompt)
    return translated_prompt

class PromptRequest(BaseModel):
    prompt: str

#Post(번역 함수 호출 및 이미지 생성 후 메세지 및 경로 반환)
#Body가 Json 형태
@app.post("/generate-image")
def generate_image(request: PromptRequest):
    prompt = translate(request.prompt)
    image = pipe(prompt=prompt,    
                 negative_prompt="Signature, Poor body structure, Low-quality drawing, Incorrect size, Outside the edges, Unclear, Dull background, Logo, Cropped, Trimmed, Body parts separated, Uneven size, Twisted, Copy, Duplicated elements, Additional arms, fingers, hands, legs, Additional body parts, Flaw, Imperfection, Joined fingers, Unpleasant size, Identifying sign, Incorrect structure, Wrong proportion, Tacky, Poor quality, Poor clarity, Spot, Absent arms, Error, Damaged, Beyond the image, Badly drawn face, feet, Text on paper, Repulsive, Shortened, Narrow eyes, Visual plan, Arrangement, Cut off, Unpleasant, Blurry, Unattractive, Awkward position, Imaginary framework, Watermark, Creature, Expression, Unbalanced, Replicated, Additional, Fiction, Textured, Individual, Unfinished, Low resolution, Lacking components, Intense, Too bright, Edited, Scary, Outline, Drawing, Unreal, Dim, Virtual engine",                  
                 height=480,
                 width=640, 
                 num_inference_steps = 60, 
                 cfg_scale = 4,
                 sampler = "Euler"
                ).images[0]
    output_path = "outputs(image)/result.png"
    image.save(output_path)
    return {"message":"OK", "file_path": output_path}
