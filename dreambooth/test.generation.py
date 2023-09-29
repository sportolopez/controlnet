from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

pipe.unet.load_attn_procs("D:/diffusers/examples/dreambooth/path-to-save-model/pytorch_lora_weights.safetensors")

image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]

image.save("dog-bucket.png")