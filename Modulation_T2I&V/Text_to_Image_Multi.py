from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers.utils import export_to_video
from PIL import Image
import os
import time
import torch
import gc
import shared
from contextlib import contextmanager
from Translate import translate, init_translation_model
from model_manager.loader import pipeline_manager

# 출력 디렉터리 생성
os.makedirs("outputs", exist_ok=True)

# 요청 데이터 모델
class SimpleImageRequest(BaseModel):
    model_name: str
    prompt: str

class SimpleVideoRequest(BaseModel):
    model_name: str
    prompt: str

@contextmanager
def memory_efficient_generation():
    """메모리 효율적인 생성을 위한 컨텍스트 매니저"""
    initial_memory = shared.memory_monitor.get_gpu_memory_usage()
    print(f"[MEMORY] 생성 시작 - GPU 메모리 사용률: {initial_memory:.2%}")
    
    try:
        # 생성 전 메모리 정리
        if shared.memory_monitor.should_cleanup():
            shared.memory_monitor.cleanup_memory()
        yield
    finally:
        # 생성 후 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        final_memory = shared.memory_monitor.get_gpu_memory_usage()
        print(f"[MEMORY] 생성 완료 - GPU 메모리 사용률: {final_memory:.2%}")

def get_generation_params(model_name: str, translated_prompt: str):
    """동적 파라미터 조정이 포함된 이미지 생성 파라미터"""
    base_params = {
        "prompt": translated_prompt,
        "negative_prompt": "blurry, low quality, distorted, deformed, bad anatomy",
        "height": 512,
        "width": 512,
        "num_inference_steps": 20,
        "guidance_scale": 5,
    }
    
    # 메모리 부족 시 해상도 조정
    if shared.memory_monitor.is_low_memory():
        print("[MEMORY] 높은 메모리 사용률 감지 - 저해상도 모드로 전환")
        base_params["height"] = 384
        base_params["width"] = 384
        base_params["num_inference_steps"] = 15
    
    return base_params

def get_video_generation_params(model_name: str, translated_prompt: str):
    """동적 파라미터 조정이 포함된 비디오 생성 파라미터"""
    base_params = {
        "prompt": translated_prompt,
        "negative_prompt": "blurry, low quality, distorted, deformed",
        "height": 512,
        "width": 512,
        "num_frames": 33,
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
    }
    
    return base_params

def sync_generate_image(pipe, generation_params: dict, output_path: str):
    """메모리 최적화된 이미지 생성"""
    try:
        # 모델을 평가 모드로 설정
        if hasattr(pipe, 'unet'):
            pipe.unet.eval()
        if hasattr(pipe, 'vae'):
            pipe.vae.eval()

        with memory_efficient_generation():
            with torch.no_grad():
                # 메모리 체크 및 RuntimeError 제거
                # 메모리 사용량 로깅만 수행
                memory_usage = shared.memory_monitor.get_gpu_memory_usage()
                print(f"[MEMORY] 현재 GPU 메모리 사용률: {memory_usage:.2%}")
                
                image = pipe(**generation_params).images[0]
                image.save(output_path)
                # 이미지 객체 즉시 해제
                del image
                return output_path
                
    except torch.cuda.OutOfMemoryError as e:
        print(f"[MEMORY] GPU 메모리 부족 - 긴급 정리 수행")
        shared.memory_monitor.cleanup_memory(force=True)
        raise RuntimeError(f"GPU 메모리 부족으로 이미지 생성 실패: {str(e)}")
    except Exception as e:
        print(f"[GENERATION] 이미지 생성 중 오류: {e}")
        raise e

def sync_generate_video(pipe, generation_params: dict, output_path: str):
    """메모리 최적화된 비디오 생성"""
    try:
        # 모델을 평가 모드로 설정
        if hasattr(pipe, 'transformer'):
            pipe.transformer.eval()
        elif hasattr(pipe, 'unet'):
            pipe.unet.eval()
        if hasattr(pipe, 'vae'):
            pipe.vae.eval()

        with memory_efficient_generation():
            with torch.no_grad():
                # 메모리 체크 및 RuntimeError 제거
                # 메모리 사용량 로깅만 수행
                memory_usage = shared.memory_monitor.get_gpu_memory_usage()
                print(f"[MEMORY] 현재 GPU 메모리 사용률: {memory_usage:.2%}")
                
                result = pipe(**generation_params)
                video_frames = result.frames[0]
                # 비디오 내보내기
                export_to_video(video_frames, output_path, fps=15)
                # 메모리 해제
                del result, video_frames
                return output_path
                
    except torch.cuda.OutOfMemoryError as e:
        print(f"[MEMORY] GPU 메모리 부족 - 긴급 정리 수행")
        shared.memory_monitor.cleanup_memory(force=True)
        raise RuntimeError(f"GPU 메모리 부족으로 비디오 생성 실패: {str(e)}")
    except Exception as e:
        print(f"[GENERATION] 비디오 생성 중 오류: {e}")
        raise e

def process_image_generation_request(request: SimpleImageRequest, task_id: str):
    """이미지 생성 요청 처리"""
    print(f"[GENERATION] [{task_id}] Prompt: {request.prompt}")
    
    try:
        translated_prompt = translate(request.prompt)
        
        # 컨텍스트 매니저를 사용한 안전한 파이프라인 관리
        with pipeline_manager.acquire_pipeline_context(request.model_name) as pipe:
            params = get_generation_params(request.model_name, translated_prompt)
            filename = f"result_{int(time.time())}_{request.model_name}_{task_id}.png"
            path = f"outputs/{filename}"
            sync_generate_image(pipe, params, path)
            
            return {"message": "OK", "image_path": path, "task_id": task_id}
            
    except Exception as e:
        print(f"[GENERATION] [{task_id}] 처리 실패: {e}")
        raise e

def process_video_generation_request(request: SimpleVideoRequest, task_id: str):
    """비디오 생성 요청 처리"""
    print(f"[VIDEO] [{task_id}] Prompt: {request.prompt}")
    
    try:
        translated_prompt = translate(request.prompt)
        
        # 컨텍스트 매니저를 사용한 안전한 파이프라인 관리
        with pipeline_manager.acquire_pipeline_context(request.model_name) as pipe:
            params = get_video_generation_params(request.model_name, translated_prompt)
            filename = f"video_{int(time.time())}_{request.model_name}_{task_id}.mp4"
            path = f"outputs/{filename}"
            sync_generate_video(pipe, params, path)
            
            return {"message": "OK", "image_path": path, "task_id": task_id}
            
    except Exception as e:
        print(f"[VIDEO] [{task_id}] 처리 실패: {e}")
        raise e

# FastAPI 앱 구성
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
    print("[STARTUP] 모델 풀 초기화 완료")
    print(f"[STARTUP] 시스템 정보: {shared.get_system_info()}")

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

@app.get("/memory-status")
def get_memory_status():
    """GPU 메모리 상태 조회"""
    memory_info = shared.memory_monitor.get_gpu_memory_info()
    model_info = pipeline_manager.get_model_info() if hasattr(pipeline_manager, 'get_model_info') else {}
    
    return {
        "gpu_memory": memory_info,
        "model_info": model_info,
        "cleanup_recommended": shared.memory_monitor.should_cleanup(),
        "system_info": shared.get_system_info()
    }

@app.post("/cleanup-memory")
def cleanup_memory():
    """수동 메모리 정리"""
    success = shared.memory_monitor.cleanup_memory(force=True)
    memory_info = shared.memory_monitor.get_gpu_memory_info()
    
    return {
        "message": "메모리 정리 완료" if success else "메모리 정리 실패",
        "success": success,
        "current_memory": memory_info
    }

@app.post("/generate-image")
def generate_image(request: SimpleImageRequest):
    task_id = f"task_{int(time.time() * 1000)}"
    
    if request.model_name not in ["SD_1.5", "SD_3.5M"]:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 모델: {request.model_name}"
        )
    
    try:
        return process_image_generation_request(request, task_id)
    except Exception as e:
        print(f"[API] 이미지 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-video")
def generate_video(request: SimpleVideoRequest):
    task_id = f"video_task_{int(time.time() * 1000)}"
    
    if request.model_name != "Wan_1.3B":
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 비디오 모델: {request.model_name}"
        )
    
    try:
        return process_video_generation_request(request, task_id)
    except Exception as e:
        print(f"[API] 비디오 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = f"outputs/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    
    if filename.endswith(".mp4"):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    elif filename.endswith(".gif"):
        return FileResponse(file_path, media_type="image/gif", filename=filename)
    else:
        return FileResponse(file_path, media_type="image/png", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Text_to_Image_Multi:app", host="127.0.0.1", port=8000, reload=True)
