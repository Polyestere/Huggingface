from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from Translate import translate, init_translation_model
import os
import asyncio
from collections import defaultdict
import time
import shared
import torch
import gc
from ModelLoad import model_load_multi, acquire_pipeline, release_pipeline

os.makedirs("outputs", exist_ok=True)

class ImageRequest(BaseModel):
    model_name: str
    prompt: str
    height: int = 512
    width: int = 512

active_tasks = set()  # 활성 태스크 추적
MAX_CONCURRENT_GENERATIONS = 2  # 동시에 실행할 최대 이미지 생성 태스크 수
generation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

# 동일 모델 병렬처리를 위한 설정
PARALLEL_SAME_MODEL = True  # 동일 모델 병렬처리 여부
MAX_SAME_MODEL_INSTANCES = 2  # 동일 모델 최대 인스턴스 수

# 모델별 세마포어 - 동일 모델 병렬처리 허용
if PARALLEL_SAME_MODEL:
    model_semaphores = {
        "SD_1.5": asyncio.Semaphore(MAX_SAME_MODEL_INSTANCES),
        "SD_3.5M": asyncio.Semaphore(MAX_SAME_MODEL_INSTANCES)
    }
else:
    model_semaphores = {
        "SD_1.5": asyncio.Semaphore(1),
        "SD_3.5M": asyncio.Semaphore(1)
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 번역 모델 초기화
    init_translation_model()
    
    # 모델 풀 초기화
    for model_name in ["SD_3.5M", "SD_1.5"]:
        if PARALLEL_SAME_MODEL:
            # 동일 모델 병렬처리용 - 여러 인스턴스 생성
            model_load_multi(model_name, num_instances=MAX_SAME_MODEL_INSTANCES)
        else:
            # 단일 인스턴스만 생성
            model_load_multi(model_name, num_instances=1)
    
    print(f"[PRELOAD] Model pools initialized (parallel_same_model={PARALLEL_SAME_MODEL})")
    yield
    
    # 정리 작업
    if active_tasks:
        print(f"[CLEANUP] Waiting for {len(active_tasks)} active tasks to complete...")
        await asyncio.gather(*active_tasks, return_exceptions=True)
        print("[CLEANUP] All tasks completed")

app = FastAPI(lifespan=lifespan)

def get_generation_params(model_name: str, translated_prompt: str, height: int, width: int):
    """생성 파라미터 설정"""
    params = {
        "prompt": translated_prompt,
        "negative_prompt": "blurry, low quality, distorted, deformed, bad anatomy",
        "height": height,
        "width": width,
        "num_inference_steps": 20,
        "guidance_scale": 5
    }
    return params

def sync_generate_image(pipe, generation_params: dict, output_path: str):
    """동기적으로 이미지 생성"""
    try:
        # 모델을 평가 모드로 설정
        if hasattr(pipe, 'unet'):
            pipe.unet.eval()
        
        # CUDA 컨텍스트 확인
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 이미지 생성
        with torch.no_grad():
            image = pipe(**generation_params).images[0]
        
        # 이미지 저장
        image.save(output_path)
        
        # 메모리 정리
        del image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_path
        
    except Exception as e:
        print(f"[GENERATION] Error in sync_generate_image: {e}")
        raise e

@app.get("/status")
async def get_status():
    """현재 처리 중인 태스크 상태를 반환합니다."""
    return {
        "active_tasks": len(active_tasks),
        "models_loaded": len(shared.cached_models) if 'shared' in globals() else 0,
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
        "parallel_same_model": PARALLEL_SAME_MODEL,
        "max_same_model_instances": MAX_SAME_MODEL_INSTANCES
    }

@app.get("/models")
async def get_available_models():
    models_info = [
        {
            "model_name": "SD_3.5M",
            "estimated_time_seconds": "60초",
            "max_parallel_instances": MAX_SAME_MODEL_INSTANCES if PARALLEL_SAME_MODEL else 1
        },
        {
            "model_name": "SD_1.5", 
            "estimated_time_seconds": "50초",
            "max_parallel_instances": MAX_SAME_MODEL_INSTANCES if PARALLEL_SAME_MODEL else 1
        }
    ]
    
    return {
        "available_models": models_info,
        "parallel_same_model_enabled": PARALLEL_SAME_MODEL
    }

async def process_generation_request(request: ImageRequest, task_id: str):
    """이미지 생성 요청 처리"""
    # 전체 생성 세마포어 획득
    async with generation_semaphore:
        # 모델별 세마포어 획득
        model_semaphore = model_semaphores.get(request.model_name)
        if not model_semaphore:
            raise ValueError(f"Unknown model: {request.model_name}")
        
        async with model_semaphore:
            try:
                print(f"[GENERATION] [{task_id}] Starting generation with {request.model_name}")
                print(f"[GENERATION] [{task_id}] Original prompt: {request.prompt}")

                # 번역 처리
                translated_prompt = translate(request.prompt)
                print(f"[GENERATION] [{task_id}] Translated prompt: {translated_prompt}")

                # 파이프라인 획득 (타임아웃 설정)
                pipe = await asyncio.to_thread(acquire_pipeline, request.model_name, timeout=120.0)
                
                try:
                    # 생성 파라미터 설정
                    generation_params = get_generation_params(
                        request.model_name,
                        translated_prompt,
                        request.height,
                        request.width
                    )

                    # 출력 파일 설정
                    timestamp = int(time.time())
                    filename = f"result_{timestamp}_{request.model_name}_{task_id}.png"
                    output_path = f"outputs/{filename}"

                    print(f"[GENERATION] [{task_id}] Generating image...")

                    # 이미지 생성 (스레드에서 실행)
                    result_path = await asyncio.to_thread(
                        sync_generate_image, 
                        pipe, 
                        generation_params, 
                        output_path
                    )

                    print(f"[GENERATION] [{task_id}] Completed: {result_path}")
                    
                    return {
                        "message": "OK",
                        "output_path": result_path,
                        "task_id": task_id
                    }
                    
                finally:
                    # 파이프라인 반환
                    await asyncio.to_thread(release_pipeline, request.model_name, pipe)
                    
            except Exception as e:
                print(f"[GENERATION] [{task_id}] Error: {e}")
                raise e

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    """이미지 생성 API"""
    task_id = f"task_{int(time.time() * 1000)}_{len(active_tasks)}"
    
    # 지원되는 모델 확인
    if request.model_name not in ["SD_1.5", "SD_3.5M"]:
        return {
            "message": "ERROR", 
            "error": f"Unsupported model: {request.model_name}",
            "task_id": task_id
        }
    
    active_tasks.add(task_id)
    
    try:
        result = await process_generation_request(request, task_id)
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"[TASK] [{task_id}] Error: {error_msg}")
        
        return {
            "message": "ERROR", 
            "error": error_msg, 
            "task_id": task_id
        }
        
    finally:
        active_tasks.discard(task_id)
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@app.get("/download/{filename}")
async def download_image(filename: str):
    """생성된 이미지 다운로드"""
    file_path = f"outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png", filename=filename)
    else:
        return {"error": "File not found"}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "active_tasks": len(active_tasks),
        "parallel_same_model": PARALLEL_SAME_MODEL
    }

@app.get("/memory")
async def get_memory_info():
    """GPU 메모리 사용량 확인"""
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    return {"error": "CUDA not available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "Text_to_Image_Multi:app",
        host="127.0.0.1",
        port=8000,
        timeout_keep_alive=300,
        reload=True
    )