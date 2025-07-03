from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
import torch
import shared
import gc

def model_load(model_name: str):
    if model_name in shared.cached_models:
        pipe = shared.cached_models[model_name]
        if hasattr(pipe, 'unet'):
            pipe.unet.eval()
        print(f"Using cached model: {model_name}")
        return pipe

    print(f"Loading model: {model_name}")

    if model_name == "SD_3.5M":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.bfloat16
        )
    elif model_name == "SD_1.5":
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        torch.cuda.empty_cache()
        print(f"Model {model_name} loaded on GPU")
    else:
        print(f"Model {model_name} loaded on CPU (CUDA not available)")

    shared.cached_models[model_name] = pipe
    gc.collect()
    print(f"Model {model_name} loaded and cached successfully")
    return pipe