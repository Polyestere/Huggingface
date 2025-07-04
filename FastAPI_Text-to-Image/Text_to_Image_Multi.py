from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from Translate import translate, init_translation_model
import os
import asyncio
import time

os.makedirs("outputs", exist_ok=True)

class ImageRequest(BaseModel):
    model_name: str
    prompt: str
    height: int = 512
    width: int = 512

model_locks = defaultdict(asyncio.Lock)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_translation_model()
    yield

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

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    print(f"[GENERATION] Starting generation with {request.model_name}")
    print(f"[GENERATION] Original prompt: {request.prompt}")

    translated_prompt = translate(request.prompt)

    generation_params = get_generation_params(
        request.model_name, 
        translated_prompt, 
        request.height, 
        request.width
    )

    timestamp = int(time.time())
    filename = f"result_{timestamp}_{request.model_name}.png"
    output_path = f"outputs/{filename}"

    print(f"[GENERATION] Generating image...")

    lock = model_locks[request.model_name]
    async with lock:
        await asyncio.to_thread(sync_generate_image, pipe, generation_params, output_path)

    print(f"[GENERATION] Completed: {output_path}")

    return {
        "message": "OK",
        "output_path": output_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "Text_to_Image_Multi:app", 
        host="127.0.0.1", 
        port=8000,
        timeout_keep_alive=300,
        reload = True
    )