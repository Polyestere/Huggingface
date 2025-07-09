from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoencoderKLWan
from pydantic import BaseModel
from diffusers.utils import export_to_video
from Translate import translate, init_translation_model
import os
import time
import shared
import torch
import gc
from PIL import Image
from ModelLoad import model_load_multi, acquire_pipeline, release_pipeline

os.makedirs("outputs", exist_ok=True)

class SimpleImageRequest(BaseModel):
    model_name: str
    prompt: str

class SimpleVideoRequest(BaseModel):
    model_name: str
    prompt: str

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
        "negative_prompt": "blurry, low quality, distorted, deformed",
        "height": 256,
        "width": 256,
        "num_frames": 33,
        "num_inference_steps": 25,
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
    try:
        if hasattr(pipe, 'transformer'):
            pipe.transformer.eval()
        elif hasattr(pipe, 'unet'):
            pipe.unet.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            result = pipe(**generation_params)
            video_frames = result.frames[0]
            export_to_video(video_frames, output_path, fps=15)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output_path
    except Exception as e:
        print(f"[GENERATION] Error in sync_generate_video: {e}")
        raise e

def process_image_generation_request(request: SimpleImageRequest, task_id: str):
    print(f"[GENERATION] [{task_id}] Starting generation with {request.model_name}")
    print(f"[GENERATION] [{task_id}] Original prompt: {request.prompt}")
    translated_prompt = translate(request.prompt)
    print(f"[GENERATION] [{task_id}] Translated prompt: {translated_prompt}")
    pipe = acquire_pipeline(request.model_name, timeout=120.0)
    try:
        generation_params = get_generation_params(request.model_name, translated_prompt)
        timestamp = int(time.time())
        filename = f"result_{timestamp}_{request.model_name}_{task_id}.png"
        output_path = f"outputs/{filename}"
        print(f"[GENERATION] [{task_id}] Generating image...")
        result_path = sync_generate_image(pipe, generation_params, output_path)
        print(f"[GENERATION] [{task_id}] Completed: {result_path}")
        return {
            "message": "OK",
            "image_path": result_path,
            "task_id": task_id
        }
    finally:
        release_pipeline(request.model_name, pipe)

def process_video_generation_request(request: SimpleVideoRequest, task_id: str):
    print(f"[VIDEO] [{task_id}] Starting video generation with {request.model_name}")
    translated_prompt = translate(request.prompt)
    pipe = acquire_pipeline(request.model_name, timeout=120.0)
    try:
        generation_params = get_video_generation_params(request.model_name, translated_prompt)
        timestamp = int(time.time())
        filename = f"video_{timestamp}_{request.model_name}_{task_id}.mp4"
        output_path = f"outputs/{filename}"
        print(f"[VIDEO] [{task_id}] Generating video...")
        result_path = sync_generate_video(pipe, generation_params, output_path)
        return {
            "message": "OK",
            "image_path": result_path,
            "task_id": task_id
        }
    finally:
        release_pipeline(request.model_name, pipe)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_translation_model()
    for model_name in ["SD_3.5M", "SD_1.5", "Wan_1.3B"]:
        model_load_multi(model_name, num_instances=1)
    print(f"[PRELOAD] Model pools initialized")

@app.get("/image-models")
def get_image_models():
    models_info = [
        {"model_name": "SD_3.5M", "estimated_time": "60초"},
        {"model_name": "SD_1.5", "estimated_time": "50초"}
    ]
    return {"available_models": models_info}

@app.get("/video-models")
def get_video_models():
    models_info = [
        {"model_name": "Wan_1.3B", "estimated_time": "120초"}
    ]
    return {"available_models": models_info}

@app.post("/generate-image")
def generate_image(request: SimpleImageRequest):
    task_id = f"task_{int(time.time() * 1000)}"
    if request.model_name not in ["SD_1.5", "SD_3.5M"]:
        return {
            "message": "ERROR",
            "error": f"Unsupported model: {request.model_name}",
            "task_id": task_id
        }
    try:
        result = process_image_generation_request(request, task_id)
        return result
    except Exception as e:
        return {
            "message": "ERROR",
            "error": str(e),
            "task_id": task_id
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@app.post("/generate-video")
def generate_video(request: SimpleVideoRequest):
    task_id = f"video_task_{int(time.time() * 1000)}"
    if request.model_name not in ["Wan_1.3B"]:
        return {
            "message": "ERROR",
            "error": f"Unsupported video model: {request.model_name}",
            "task_id": task_id
        }
    try:
        result = process_video_generation_request(request, task_id)
        return result
    except Exception as e:
        return {
            "message": "ERROR",
            "error": str(e),
            "task_id": task_id
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@app.get("/download/{filename}")
def download_file(filename: str):
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
