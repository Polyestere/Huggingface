from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
# WanPipeline import 추가 (실제 사용 시 올바른 import 경로 확인 필요)
from diffusers import WanPipeline
import torch
import shared
import gc
import psutil
import logging
from queue import Queue
import threading
import copy

logger = logging.getLogger(__name__)

# 모델별 VRAM 사용량(GB) - Wan_1.3B 추가
MODEL_VRAM = {
    "SD_1.5": 5,
    "SD_3.5M": 6,
    "Wan_1.3B": 4  # Wan2.1 T2V 모델 추가
}

MAX_VRAM = 12  # 예시: 12GB

# GPU 메모리 관리를 위한 전역 락
GPU_MEMORY_LOCK = threading.Lock()

def get_gpu_memory_gb():
    """GPU 메모리 확인"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0

def get_current_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)  # GB 단위
    return 0.0

def get_max_instances(model_name):
    vram_per_instance = MODEL_VRAM[model_name]
    # 병렬 처리 축소로 최대 1개 인스턴스만 허용
    return 1

def create_model_copy(base_pipe, model_name):
    """기존 모델로부터 독립적인 복사본 생성"""
    with GPU_MEMORY_LOCK:
        try:
            print(f"[MODEL] Creating copy for {model_name}...")
            
            if model_name == "SD_3.5M":
                new_pipe = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3.5-medium",
                    torch_dtype=base_pipe.dtype,
                    use_safetensors=True
                )
            elif model_name == "SD_1.5":
                new_pipe = StableDiffusionPipeline.from_pretrained(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    torch_dtype=base_pipe.dtype,
                    use_safetensors=True
                )
            elif model_name == "Wan_1.3B":
                new_pipe = WanPipeline.from_pretrained(
                    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    torch_dtype=base_pipe.dtype,
                    use_safetensors=True
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # GPU로 이동 및 최적화 적용
            if torch.cuda.is_available():
                new_pipe = new_pipe.to("cuda")
                # 기본 최적화 설정 적용
                gpu_memory_gb = get_gpu_memory_gb()
                if gpu_memory_gb >= 12:
                    new_pipe.enable_attention_slicing()
                elif gpu_memory_gb >= 8:
                    new_pipe.enable_attention_slicing()
                    new_pipe.enable_model_cpu_offload()
                    if model_name == "SD_1.5":
                        if hasattr(new_pipe, 'enable_vae_slicing'):
                            new_pipe.enable_vae_slicing()
                else:
                    new_pipe.enable_attention_slicing()
                    new_pipe.enable_sequential_cpu_offload()
                    if hasattr(new_pipe, 'enable_vae_slicing'):
                        new_pipe.enable_vae_slicing()
                    if hasattr(new_pipe, 'enable_vae_tiling'):
                        new_pipe.enable_vae_tiling()

            torch.cuda.empty_cache()
            print(f"[MODEL] Successfully created copy for {model_name}")
            return new_pipe

        except Exception as e:
            print(f"[MODEL] Failed to create model copy: {e}")
            return base_pipe

def model_load(model_name: str):
    """단일 모델 로딩"""
    with GPU_MEMORY_LOCK:
        if model_name in shared.cached_models:
            pipe = shared.cached_models[model_name]
            if hasattr(pipe, 'unet'):
                pipe.unet.eval()
            elif hasattr(pipe, 'transformer'):
                pipe.transformer.eval()
            return pipe

        print(f"[MODEL] Loading {model_name}...")
        gpu_memory_gb = get_gpu_memory_gb()
        print(f"[MODEL] Available GPU memory: {gpu_memory_gb:.1f}GB")

        try:
            if model_name == "SD_3.5M":
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3.5-medium",
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True
                )
            elif model_name == "SD_1.5":
                pipe = StableDiffusionPipeline.from_pretrained(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            elif model_name == "Wan_1.3B":
                pipe = WanPipeline.from_pretrained(
                    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                
                # GPU 메모리에 따른 적응적 최적화
                if gpu_memory_gb >= 12:
                    print(f"[MODEL] High-end GPU detected. Using performance optimizations.")
                    pipe.enable_attention_slicing()
                    if model_name == "Wan_1.3B":
                        pipe.enable_model_cpu_offload()
                elif gpu_memory_gb >= 8:
                    print(f"[MODEL] Mid-range GPU detected. Using balanced optimizations.")
                    pipe.enable_attention_slicing()
                    pipe.enable_model_cpu_offload()
                    if model_name == "SD_1.5":
                        if hasattr(pipe, 'enable_vae_slicing'):
                            pipe.enable_vae_slicing()
                else:
                    print(f"[MODEL] Low-end GPU detected. Using memory-saving optimizations.")
                    pipe.enable_attention_slicing()
                    pipe.enable_sequential_cpu_offload()
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                    if hasattr(pipe, 'enable_vae_tiling'):
                        pipe.enable_vae_tiling()

                torch.cuda.empty_cache()
            else:
                print(f"[MODEL] No CUDA available. Using CPU mode.")

            shared.cached_models[model_name] = pipe
            gc.collect()
            print(f"[MODEL] {model_name} loaded successfully")
            return pipe

        except Exception as e:
            print(f"[MODEL] Failed to load {model_name}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 폴백 시도
            try:
                print(f"[MODEL] Attempting fallback loading for {model_name}...")
                pipe = None
                
                if model_name == "SD_3.5M":
                    pipe = StableDiffusion3Pipeline.from_pretrained(
                        "stabilityai/stable-diffusion-3.5-medium",
                        torch_dtype=torch.float16,
                        use_safetensors=True
                    )
                elif model_name == "SD_1.5":
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16,
                        use_safetensors=True
                    )
                elif model_name == "Wan_1.3B":
                    pipe = WanPipeline.from_pretrained(
                        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        torch_dtype=torch.float16,
                        use_safetensors=True
                    )
                
                if pipe is not None:
                    if torch.cuda.is_available():
                        pipe = pipe.to("cuda")
                        pipe.enable_attention_slicing()
                        torch.cuda.empty_cache()
                    
                    shared.cached_models[model_name] = pipe
                    gc.collect()
                    print(f"[MODEL] {model_name} loaded successfully with fallback")
                    return pipe
                else:
                    raise ValueError(f"Unknown model in fallback: {model_name}")
                    
            except Exception as fallback_error:
                print(f"[MODEL] Fallback also failed: {fallback_error}")
                raise fallback_error

def model_load_multi(model_name: str, num_instances: int = None):
    """VRAM 한도 내에서 모델별 파이프라인 인스턴스를 생성하여 풀에 저장"""
    if model_name in shared.pipeline_pools:
        return

    if num_instances is None:
        num_instances = get_max_instances(model_name)

    base_pipe = model_load(model_name)
    pool = Queue(maxsize=num_instances)
    pool.put(base_pipe)

    shared.pipeline_pools[model_name] = pool
    shared.pipeline_pool_locks[model_name] = threading.Lock()
    print(f"[MODEL] {model_name} pipeline pool initialized (1 instance)")

def acquire_pipeline(model_name: str, timeout: float = 60.0):
    """사용 가능한 파이프라인 인스턴스를 풀에서 가져옴"""
    if model_name not in shared.pipeline_pools:
        print(f"[MODEL] Pipeline pool for {model_name} not found, creating...")
        model_load_multi(model_name)

    pool = shared.pipeline_pools[model_name]
    try:
        pipe = pool.get(timeout=timeout)
        print(f"[MODEL] Pipeline acquired for {model_name}")
        return pipe
    except Exception as e:
        print(f"[MODEL] Failed to acquire pipeline for {model_name}: {e}")
        raise e

def release_pipeline(model_name: str, pipe):
    """사용이 끝난 파이프라인 인스턴스를 풀에 반환"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pool = shared.pipeline_pools[model_name]
        pool.put(pipe)
        print(f"[MODEL] Pipeline released for {model_name}")
    except Exception as e:
        print(f"[MODEL] Failed to release pipeline for {model_name}: {e}")

def get_model_info():
    """현재 로드된 모델 정보 반환"""
    info = {}
    for model_name, pipe in shared.cached_models.items():
        info[model_name] = {
            "loaded": True,
            "device": str(pipe.device) if hasattr(pipe, 'device') else "unknown",
            "dtype": str(pipe.dtype) if hasattr(pipe, 'dtype') else "unknown",
            "pool_size": shared.pipeline_pools[model_name].qsize() if model_name in shared.pipeline_pools else 0
        }
    return info

def clear_model_cache():
    """모델 캐시 정리"""
    with GPU_MEMORY_LOCK:
        for model_name in list(shared.cached_models.keys()):
            del shared.cached_models[model_name]
        
        for model_name in list(shared.pipeline_pools.keys()):
            pool = shared.pipeline_pools[model_name]
            while not pool.empty():
                try:
                    pipe = pool.get_nowait()
                    del pipe
                except:
                    break
            del shared.pipeline_pools[model_name]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[MODEL] Cache cleared")
