from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoencoderKLWan
from pydantic import BaseModel
from diffusers.utils import export_to_video
from Translate import translate, init_translation_model
import os
import asyncio
import time
import shared
import torch
import gc
import numpy as np
from PIL import Image
from ModelLoad import model_load_multi, acquire_pipeline, release_pipeline
import tempfile

os.makedirs("outputs", exist_ok=True)

class SimpleImageRequest(BaseModel):
    model_name: str
    prompt: str

class SimpleVideoRequest(BaseModel):
    model_name: str
    prompt: str

active_tasks = set()
generation_semaphore = asyncio.Semaphore(1)

model_semaphores = {
    "SD_1.5": asyncio.Semaphore(1),
    "SD_3.5M": asyncio.Semaphore(1),
    "Wan_1.3B": asyncio.Semaphore(1)
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_translation_model()
    
    for model_name in ["SD_3.5M", "SD_1.5", "Wan_1.3B"]:
        model_load_multi(model_name, num_instances=1)
    
    print(f"[PRELOAD] Model pools initialized")
    yield
    
    if active_tasks:
        print(f"[CLEANUP] Waiting for {len(active_tasks)} active tasks to complete...")
        await asyncio.gather(*active_tasks, return_exceptions=True)
    print("[CLEANUP] All tasks completed")

app = FastAPI(lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_generation_params(model_name: str, translated_prompt: str):
    return {
        "prompt": translated_prompt,
        "negative_prompt": "blurry, low quality, distorted, deformed, bad anatomy",
        "height": 512,
        "width": 512,
        "num_inference_steps": 20,
        "guidance_scale": 5
    }

def get_video_generation_params(model_name: str, translated_prompt: str):
    return {
        "prompt": translated_prompt,
        "negative_prompt": (
            "blurry, low quality, distorted, deformed"
        ),
        "height": 384,      # 예시 코드와 동일
        "width": 384,       # 예시 코드와 동일
        "num_frames": 45,   # 21-1=20, 4의 배수
        "num_inference_steps": 20,
        "guidance_scale": 5.0
    }

def sync_generate_image(pipe, generation_params: dict, output_path: str):
    try:
        if hasattr(pipe, 'unet'):
            pipe.unet.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            image = pipe(**generation_params).images[0]
        
        image.save(output_path)
        del image
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_path
    except Exception as e:
        print(f"[GENERATION] Error in sync_generate_image: {e}")
        raise e

def sync_generate_video(pipe, generation_params: dict, output_path: str):
    """
    Diffusers 공식 예제 스타일:
    - pipe(**params).frames[0]만 추출
    - 별도 후처리 없이 바로 export_to_video
    """
    try:
        if hasattr(pipe, 'transformer'):
            pipe.transformer.eval()
        elif hasattr(pipe, 'unet'):
            pipe.unet.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            result = pipe(**generation_params)
            # Diffusers 공식 예제처럼 바로 frames[0]만 사용
            video_frames = result.frames[0]
            export_to_video(video_frames, output_path, fps=15)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path

    except Exception as e:
        print(f"[GENERATION] Error in sync_generate_video: {e}")
        raise e


    
# 3. 간소화된 모델 정보 API
@app.get("/image-models")
async def get_image_models():
    """이미지 생성 모델 정보를 반환합니다."""
    models_info = [
        {
            "model_name": "SD_3.5M",
            "estimated_time": "60초"
        },
        {
            "model_name": "SD_1.5", 
            "estimated_time": "50초"
        }
    ]
    
    return {
        "available_models": models_info
    }

@app.get("/video-models")
async def get_video_models():
    """비디오 생성 모델 정보를 반환합니다."""
    models_info = [
        {
            "model_name": "Wan_1.3B",
            "estimated_time": "120초"
        }
    ]
    
    return {
        "available_models": models_info
    }

# 누락된 함수들 추가
async def process_image_generation_request(request: SimpleImageRequest, task_id: str):
    """이미지 생성 요청 처리"""
    async with generation_semaphore:
        model_semaphore = model_semaphores.get(request.model_name)
        if not model_semaphore:
            raise ValueError(f"Unknown model: {request.model_name}")

        async with model_semaphore:
            try:
                print(f"[GENERATION] [{task_id}] Starting generation with {request.model_name}")
                print(f"[GENERATION] [{task_id}] Original prompt: {request.prompt}")

                translated_prompt = translate(request.prompt)
                print(f"[GENERATION] [{task_id}] Translated prompt: {translated_prompt}")

                pipe = await asyncio.to_thread(acquire_pipeline, request.model_name, timeout=120.0)

                try:
                    generation_params = get_generation_params(request.model_name, translated_prompt)

                    timestamp = int(time.time())
                    filename = f"result_{timestamp}_{request.model_name}_{task_id}.png"
                    output_path = f"outputs/{filename}"

                    print(f"[GENERATION] [{task_id}] Generating image...")
                    result_path = await asyncio.to_thread(
                        sync_generate_image,
                        pipe,
                        generation_params,
                        output_path
                    )

                    print(f"[GENERATION] [{task_id}] Completed: {result_path}")
                    return {
                        "message": "OK",
                        "image_path": result_path,
                        "task_id": task_id
                    }

                finally:
                    await asyncio.to_thread(release_pipeline, request.model_name, pipe)

            except Exception as e:
                print(f"[GENERATION] [{task_id}] Error: {e}")
                raise e

async def process_video_generation_request(request: SimpleVideoRequest, task_id: str):
    """비디오 생성 요청 처리 - export_to_video 사용"""
    async with generation_semaphore:
        model_semaphore = model_semaphores.get(request.model_name)
        if not model_semaphore:
            raise ValueError(f"Unknown model: {request.model_name}")
        async with model_semaphore:
            try:
                print(f"[VIDEO] [{task_id}] Starting video generation with {request.model_name}")
                translated_prompt = translate(request.prompt)
                pipe = await asyncio.to_thread(acquire_pipeline, request.model_name, timeout=120.0)
                try:
                    generation_params = get_video_generation_params(request.model_name, translated_prompt)
                    timestamp = int(time.time())
                    filename = f"video_{timestamp}_{request.model_name}_{task_id}.mp4"
                    output_path = f"outputs/{filename}"
                    print(f"[VIDEO] [{task_id}] Generating video...")
                    result_path = await asyncio.to_thread(
                        sync_generate_video,
                        pipe,
                        generation_params,
                        output_path
                    )
                    return {
                        "message": "OK",
                        "image_path": result_path,
                        "task_id": task_id
                    }
                finally:
                    await asyncio.to_thread(release_pipeline, request.model_name, pipe)
            except Exception as e:
                print(f"[VIDEO] [{task_id}] Error: {e}")
                raise e


@app.post("/generate-image")
async def generate_image(request: SimpleImageRequest):
    task_id = f"task_{int(time.time() * 1000)}"
    
    if request.model_name not in ["SD_1.5", "SD_3.5M"]:
        return {
            "message": "ERROR",
            "error": f"Unsupported model: {request.model_name}",
            "task_id": task_id
        }

    active_tasks.add(task_id)
    try:
        result = await process_image_generation_request(request, task_id)
        return result
    except Exception as e:
        return {
            "message": "ERROR",
            "error": str(e),
            "task_id": task_id
        }
    finally:
        active_tasks.discard(task_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@app.post("/generate-video")
async def generate_video(request: SimpleVideoRequest):
    task_id = f"video_task_{int(time.time() * 1000)}"
    if request.model_name not in ["Wan_1.3B"]:
        return {
            "message": "ERROR",
            "error": f"Unsupported video model: {request.model_name}",
            "task_id": task_id
        }
    active_tasks.add(task_id)
    try:
        result = await process_video_generation_request(request, task_id)
        return result
    except Exception as e:
        return {
            "message": "ERROR",
            "error": str(e),
            "task_id": task_id
        }
    finally:
        active_tasks.discard(task_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@app.get("/download/{filename}")
async def download_file(filename: str):
    """생성된 파일 다운로드 - MP4 지원 추가"""
    file_path = f"outputs/{filename}"
    if os.path.exists(file_path):
        if filename.endswith('.mp4'):
            return FileResponse(file_path, media_type="video/mp4", filename=filename)
        elif filename.endswith('.gif'):
            return FileResponse(file_path, media_type="image/gif", filename=filename)
        else:
            return FileResponse(file_path, media_type="image/png", filename=filename)
    else:
        return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "Text_to_Image_Multi:app",
        host="127.0.0.1",
        port=8000,
        timeout_keep_alive=300,
        reload=True
    )
