from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
import torch
import shared
import gc
import psutil
import logging
from queue import Queue
import threading
import copy

logger = logging.getLogger(__name__)

# 모델별 VRAM 사용량(GB)
MODEL_VRAM = {
    "SD_1.5": 5,
    "SD_3.5M": 6
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
    # 병렬 처리를 위해 최대 인스턴스 수를 제한
    return min(2, max(1, int(MAX_VRAM // vram_per_instance)))

def create_model_copy(base_pipe, model_name):
    """기존 모델로부터 독립적인 복사본 생성"""
    with GPU_MEMORY_LOCK:
        try:
            print(f"[MODEL] Creating copy for {model_name}...")
            
            # 방법 1: 원본 모델 경로에서 새로 로드 (권장)
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
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # GPU로 이동 및 최적화 적용
            if torch.cuda.is_available():
                new_pipe = new_pipe.to("cuda")
                
                # 기본 파이프라인과 같은 최적화 설정 적용
                gpu_memory_gb = get_gpu_memory_gb()
                
                if gpu_memory_gb >= 12:  # 12GB 이상 - 성능 우선
                    new_pipe.enable_attention_slicing()
                    
                elif gpu_memory_gb >= 8:  # 8-12GB - 균형
                    new_pipe.enable_attention_slicing()
                    new_pipe.enable_model_cpu_offload()
                    
                    if model_name == "SD_1.5":
                        if hasattr(new_pipe, 'enable_vae_slicing'):
                            new_pipe.enable_vae_slicing()
                    
                elif gpu_memory_gb >= 6:  # 6-8GB - 메모리 절약 우선
                    new_pipe.enable_attention_slicing()
                    new_pipe.enable_sequential_cpu_offload()
                    
                    if hasattr(new_pipe, 'enable_vae_slicing'):
                        new_pipe.enable_vae_slicing()
                    if hasattr(new_pipe, 'enable_vae_tiling'):
                        new_pipe.enable_vae_tiling()
                        
                else:  # 6GB 미만 - 극도로 보수적
                    new_pipe.enable_attention_slicing()
                    new_pipe.enable_sequential_cpu_offload()
                    
                    if hasattr(new_pipe, 'enable_vae_slicing'):
                        new_pipe.enable_vae_slicing()
                    if hasattr(new_pipe, 'enable_vae_tiling'):
                        new_pipe.enable_vae_tiling()
                
                # 메모리 정리
                torch.cuda.empty_cache()
            
            print(f"[MODEL] Successfully created copy for {model_name}")
            return new_pipe
            
        except Exception as e:
            print(f"[MODEL] Failed to create model copy: {e}")
            
            # 방법 2: 컴포넌트 기반 복사 시도
            try:
                print(f"[MODEL] Attempting component-based copy for {model_name}...")
                
                # 필수 컴포넌트 확인
                required_components = {}
                
                # 기본 컴포넌트들
                for comp_name in ['vae', 'tokenizer', 'text_encoder', 'scheduler', 'unet']:
                    if hasattr(base_pipe, comp_name):
                        component = getattr(base_pipe, comp_name)
                        if component is not None:
                            required_components[comp_name] = component
                
                # 선택적 컴포넌트들 (있으면 추가)
                for comp_name in ['safety_checker', 'feature_extractor', 'image_encoder']:
                    if hasattr(base_pipe, comp_name):
                        component = getattr(base_pipe, comp_name)
                        if component is not None:
                            required_components[comp_name] = component
                
                print(f"[MODEL] Available components: {list(required_components.keys())}")
                
                # 파이프라인 타입에 따른 생성
                if model_name == "SD_3.5M":
                    new_pipe = StableDiffusion3Pipeline(**required_components)
                elif model_name == "SD_1.5":
                    new_pipe = StableDiffusionPipeline(**required_components)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # GPU로 이동
                if torch.cuda.is_available():
                    new_pipe = new_pipe.to("cuda")
                    torch.cuda.empty_cache()
                
                print(f"[MODEL] Successfully created component-based copy for {model_name}")
                return new_pipe
                
            except Exception as comp_error:
                print(f"[MODEL] Component-based copy also failed: {comp_error}")
                
                # 방법 3: 최후의 수단 - 기존 파이프라인 공유 (위험)
                print(f"[MODEL] WARNING: Falling back to shared instance for {model_name}")
                print(f"[MODEL] This may cause CUDA errors with parallel processing!")
                return base_pipe

def model_load(model_name: str):
    """단일 모델 로딩"""
    with GPU_MEMORY_LOCK:
        if model_name in shared.cached_models:
            pipe = shared.cached_models[model_name]
            if hasattr(pipe, 'unet'):
                pipe.unet.eval()
            return pipe

        print(f"[MODEL] Loading {model_name}...")
        
        # GPU 메모리 확인
        gpu_memory_gb = get_gpu_memory_gb()
        print(f"[MODEL] Available GPU memory: {gpu_memory_gb:.1f}GB")
        
        try:
            # 기본 모델 로딩
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
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                
                # GPU 메모리에 따른 적응적 최적화
                if gpu_memory_gb >= 12:  # 12GB 이상 - 성능 우선
                    print(f"[MODEL] High-end GPU detected. Using performance optimizations.")
                    pipe.enable_attention_slicing()
                    
                elif gpu_memory_gb >= 8:  # 8-12GB - 균형
                    print(f"[MODEL] Mid-range GPU detected. Using balanced optimizations.")
                    pipe.enable_attention_slicing()
                    pipe.enable_model_cpu_offload()
                    
                    if model_name == "SD_1.5":
                        if hasattr(pipe, 'enable_vae_slicing'):
                            pipe.enable_vae_slicing()
                    
                elif gpu_memory_gb >= 6:  # 6-8GB - 메모리 절약 우선
                    print(f"[MODEL] Low-end GPU detected. Using memory-saving optimizations.")
                    pipe.enable_attention_slicing()
                    pipe.enable_sequential_cpu_offload()
                    
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                    if hasattr(pipe, 'enable_vae_tiling'):
                        pipe.enable_vae_tiling()
                        
                else:  # 6GB 미만 - 극도로 보수적
                    print(f"[MODEL] Very low GPU memory. Using conservative settings.")
                    pipe.enable_attention_slicing()
                    pipe.enable_sequential_cpu_offload()
                    
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                    if hasattr(pipe, 'enable_vae_tiling'):
                        pipe.enable_vae_tiling()
                
                # 메모리 정리
                torch.cuda.empty_cache()
                
            else:
                print(f"[MODEL] No CUDA available. Using CPU mode.")
            
            # 캐시에 저장
            shared.cached_models[model_name] = pipe
            gc.collect()
            
            print(f"[MODEL] {model_name} loaded successfully")
            return pipe
            
        except Exception as e:
            print(f"[MODEL] Failed to load {model_name}: {e}")
            
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 폴백 시도
            try:
                print(f"[MODEL] Attempting fallback loading for {model_name}...")
                
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
                
                if torch.cuda.is_available():
                    pipe = pipe.to("cuda")
                    pipe.enable_attention_slicing()
                    torch.cuda.empty_cache()
                
                shared.cached_models[model_name] = pipe
                gc.collect()
                print(f"[MODEL] {model_name} loaded successfully with fallback")
                return pipe
                
            except Exception as fallback_error:
                print(f"[MODEL] Fallback also failed: {fallback_error}")
                raise fallback_error

def model_load_multi(model_name: str, num_instances: int = None):
    """
    VRAM 한도 내에서 모델별 파이프라인 인스턴스를 생성하여 풀에 저장
    병렬 처리를 위해 각 인스턴스를 독립적으로 생성
    """
    if model_name in shared.pipeline_pools:
        return  # 이미 풀 생성됨
    
    if num_instances is None:
        num_instances = get_max_instances(model_name)
    
    # 첫 번째 인스턴스 로드
    base_pipe = model_load(model_name)
    
    pool = Queue(maxsize=num_instances)
    
    # 첫 번째 인스턴스를 풀에 추가
    pool.put(base_pipe)
    
    # 추가 인스턴스들을 독립적으로 생성
    successful_instances = 1
    for i in range(1, num_instances):
        try:
            print(f"[MODEL] Creating additional instance {i+1}/{num_instances} for {model_name}")
            
            # 독립적인 모델 인스턴스 생성
            additional_pipe = create_model_copy(base_pipe, model_name)
            
            # 복사본이 실제로 다른 인스턴스인지 확인
            if additional_pipe is not base_pipe:
                pool.put(additional_pipe)
                successful_instances += 1
                print(f"[MODEL] Successfully created instance {i+1} for {model_name}")
            else:
                print(f"[MODEL] WARNING: Instance {i+1} is shared (not independent)")
                # 공유 인스턴스라도 풀에 추가 (병렬 처리 위험)
                pool.put(additional_pipe)
                successful_instances += 1
                
        except Exception as e:
            print(f"[MODEL] Failed to create additional instance {i+1}: {e}")
            # 실패 시 더 이상 인스턴스 생성 시도하지 않음
            break
    
    shared.pipeline_pools[model_name] = pool
    shared.pipeline_pool_locks[model_name] = threading.Lock()
    
    print(f"[MODEL] {model_name} pipeline pool initialized ({successful_instances} instances)")
    
    # 병렬 처리 가능 여부 확인
    if successful_instances < num_instances:
        print(f"[MODEL] WARNING: Only {successful_instances}/{num_instances} instances created")
        print(f"[MODEL] Parallel processing may be limited for {model_name}")

def acquire_pipeline(model_name: str, timeout: float = 60.0):
    """
    사용 가능한 파이프라인 인스턴스를 풀에서 가져옴
    """
    if model_name not in shared.pipeline_pools:
        print(f"[MODEL] Pipeline pool for {model_name} not found, creating...")
        model_load_multi(model_name)
    
    pool = shared.pipeline_pools[model_name]
    try:
        pipe = pool.get(timeout=timeout)
        print(f"[MODEL] Pipeline acquired for {model_name} (pool size: {pool.qsize()})")
        return pipe
    except Exception as e:
        print(f"[MODEL] Failed to acquire pipeline for {model_name}: {e}")
        raise e

def release_pipeline(model_name: str, pipe):
    """
    사용이 끝난 파이프라인 인스턴스를 풀에 반환
    """
    try:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pool = shared.pipeline_pools[model_name]
        pool.put(pipe)
        print(f"[MODEL] Pipeline released for {model_name} (pool size: {pool.qsize()})")
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
        
        # 파이프라인 풀도 정리
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