from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
import torch
import shared
import gc
import psutil
import logging

logger = logging.getLogger(__name__)

def get_gpu_memory_gb():
    """GPU 메모리 확인"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0

def model_load(model_name: str):
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
        # 기본 모델 로딩 (variant 없이)
        if model_name == "SD_3.5M":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-medium",
                torch_dtype=torch.bfloat16,
                use_safetensors=True
                # variant="fp16" 제거 - 존재하지 않을 수 있음
            )
        elif model_name == "SD_1.5":
            pipe = StableDiffusionPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                use_safetensors=True
                # variant="fp16" 제거 - 안정성 우선
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            
            # GPU 메모리에 따른 적응적 최적화
            if gpu_memory_gb >= 12:  # 12GB 이상 - 성능 우선
                print(f"[MODEL] High-end GPU detected. Using performance optimizations.")
                pipe.enable_attention_slicing()
                
                # torch.compile 시도 (실패해도 계속 진행)
                if hasattr(torch, 'compile') and model_name == "SD_1.5":  # SD 1.5에서만 시도
                    try:
                        if hasattr(pipe, 'unet'):
                            print(f"[MODEL] Attempting UNet compilation...")
                            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
                            print(f"[MODEL] UNet compiled successfully")
                    except Exception as e:
                        print(f"[MODEL] Compilation failed (continuing without): {e}")
                
            elif gpu_memory_gb >= 8:  # 8-12GB - 균형
                print(f"[MODEL] Mid-range GPU detected. Using balanced optimizations.")
                pipe.enable_attention_slicing()
                pipe.enable_model_cpu_offload()  # 메모리 절약
                
                # VAE 최적화 (SD 1.5에서만)
                if model_name == "SD_1.5":
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                
            elif gpu_memory_gb >= 6:  # 6-8GB - 메모리 절약 우선
                print(f"[MODEL] Low-end GPU detected. Using memory-saving optimizations.")
                pipe.enable_attention_slicing()
                pipe.enable_sequential_cpu_offload()  # 강력한 메모리 절약
                
                # VAE 최적화
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
        
        # 폴백 시도 (더 보수적인 설정)
        try:
            print(f"[MODEL] Attempting fallback loading for {model_name}...")
            
            if model_name == "SD_3.5M":
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3.5-medium",
                    torch_dtype=torch.float16,  # bfloat16 대신 float16 시도
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
                # 최소한의 최적화만 적용
                pipe.enable_attention_slicing()
                torch.cuda.empty_cache()
            
            shared.cached_models[model_name] = pipe
            gc.collect()
            print(f"[MODEL] {model_name} loaded successfully with fallback")
            return pipe
            
        except Exception as fallback_error:
            print(f"[MODEL] Fallback also failed: {fallback_error}")
            raise fallback_error


def get_model_info():
    """현재 로드된 모델 정보 반환"""
    info = {}
    for model_name, pipe in shared.cached_models.items():
        info[model_name] = {
            "loaded": True,
            "device": str(pipe.device) if hasattr(pipe, 'device') else "unknown",
            "dtype": str(pipe.dtype) if hasattr(pipe, 'dtype') else "unknown"
        }
    return info


def clear_model_cache():
    """모델 캐시 정리"""
    for model_name in list(shared.cached_models.keys()):
        del shared.cached_models[model_name]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[MODEL] Cache cleared")