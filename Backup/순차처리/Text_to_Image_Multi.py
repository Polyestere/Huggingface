from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from Translate import translate, init_translation_model
import os
import asyncio
from collections import defaultdict
import time


os.makedirs("outputs", exist_ok=True)


class ImageRequest(BaseModel):
    model_name: str
    prompt: str
    height: int = 512
    width: int = 512


model_locks = defaultdict(asyncio.Lock)
active_tasks = set()  # 활성 태스크 추적


async def process_generation_request(request: ImageRequest, task_id: str):
    """실제 이미지 생성 처리 로직"""
    try:
        print(f"[GENERATION] [{task_id}] Starting generation with {request.model_name}")
        print(f"[GENERATION] [{task_id}] Original prompt: {request.prompt}")

        translated_prompt = translate(request.prompt)
        print(f"[GENERATION] [{task_id}] Translated prompt: {translated_prompt}")

        from ModelLoad import model_load
        pipe = model_load(request.model_name)

        generation_params = get_generation_params(
            request.model_name,
            translated_prompt,
            request.height,
            request.width
        )

        timestamp = int(time.time())
        filename = f"result_{timestamp}_{request.model_name}_{task_id}.png"
        output_path = f"outputs/{filename}"

        print(f"[GENERATION] [{task_id}] Generating image...")

        # 모델별 락을 사용하여 동일 모델에 대한 동시 실행 제한
        lock = model_locks[request.model_name]
        async with lock:
            await asyncio.to_thread(sync_generate_image, pipe, generation_params, output_path)

        print(f"[GENERATION] [{task_id}] Completed: {output_path}")

        return {
            "message": "OK",
            "output_path": output_path,
            "task_id": task_id
        }
    except Exception as e:
        print(f"[GENERATION] [{task_id}] Error: {e}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 번역 모델 초기화
    init_translation_model()
    
    # 모든 이미지 생성 모델 사전 로딩
    print("[PRELOAD] Starting model preloading...")
    from ModelLoad import model_load
    
    available_models = ["SD_3.5M", "SD_1.5"]
    for model_name in available_models:
        try:
            print(f"[PRELOAD] Preloading {model_name}...")
            model_load(model_name)
            print(f"[PRELOAD] {model_name} preloaded successfully")
        except Exception as e:
            print(f"[PRELOAD] Failed to preload {model_name}: {e}")
    
    print("[PRELOAD] Model preloading completed")
    
    yield
    
    # 정리 작업 - 모든 활성 태스크 완료 대기
    if active_tasks:
        print(f"[CLEANUP] Waiting for {len(active_tasks)} active tasks to complete...")
        await asyncio.gather(*active_tasks, return_exceptions=True)
        print("[CLEANUP] All tasks completed")


app = FastAPI(lifespan=lifespan)


def get_generation_params(model_name: str, translated_prompt: str, height: int, width: int):
    return {
        "prompt": translated_prompt,
        "negative_prompt": "blurry, low quality, distorted, deformed, bad anatomy",
        "height": height,
        "width": width,
        "num_inference_steps": 25,
        "guidance_scale": 7.6
    }


def sync_generate_image(pipe, generation_params: dict, output_path: str):
    if hasattr(pipe, 'unet'):
        pipe.unet.eval()
    image = pipe(**generation_params).images[0]
    image.save(output_path)
    return output_path


@app.get("/status")
async def get_status():
    """현재 처리 중인 태스크 상태를 반환합니다."""
    return {
        "active_tasks": len(active_tasks),
        "models_loaded": len(shared.cached_models) if 'shared' in globals() else 0
    }


@app.get("/models")
async def get_available_models():
    models_info = [
        {
            "model_name": "SD_3.5M",
            "estimated_time_seconds": "60초"
        },
        {
            "model_name": "SD_1.5", 
            "estimated_time_seconds": "35초"
        }
    ]
    
    return {
        "available_models": models_info
    }


@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    """이미지 생성 요청을 비동기적으로 처리합니다."""
    # 고유한 태스크 ID 생성
    task_id = f"task_{int(time.time() * 1000)}_{len(active_tasks)}"
    
    # 백그라운드 태스크 생성
    async def task_wrapper():
        try:
            result = await process_generation_request(request, task_id)
            return result
        except Exception as e:
            print(f"[TASK] [{task_id}] Error: {e}")
            raise e
        finally:
            # 태스크 완료 시 활성 태스크 목록에서 제거
            active_tasks.discard(task)
    
    # 태스크 생성 및 추가
    task = asyncio.create_task(task_wrapper())
    active_tasks.add(task)
    
    print(f"[TASK] [{task_id}] Task created. Active tasks: {len(active_tasks)}")
    
    # 태스크 완료 대기
    result = await task
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "Text_to_Image_Multi:app",
        host="127.0.0.1",
        port=8000,
        timeout_keep_alive=300,
        reload=True
    )