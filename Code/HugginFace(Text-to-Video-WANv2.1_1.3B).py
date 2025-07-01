#https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Terminal Version
# git clone https://github.com/Wan-Video/Wan2.1.git
# cd C:\Users\user\Desktop\Hugginface\Code\Wan2.1
# pip install -r requirements.txt
# pip install "huggingface_hub[cli]"
# huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ./Wan2.1-T2V-1.3B-Diffusers
# python generate.py  --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --sample_shift 8 --sample_guide_scale 6 --prompt "a video of an duck floating on the river"


# Diffusers Version
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "a duck floating on the river"
negative_prompt = "blurred detales"
# negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=640,
    num_frames=44,
    num_inference_steps = 10,
    guidance_scale=5.0
).frames[0]
export_to_video(output, "output.mp4", fps=15)