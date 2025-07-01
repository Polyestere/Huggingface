#https://huggingface.co/CompVis/stable-diffusion-v1-4

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

prompt = "a photo of an duck floating on the river"
image = pipe(prompt).images[0]

image.save("duck_floating_on_the_river_std_v1.4-2.png")