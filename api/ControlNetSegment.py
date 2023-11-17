import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDPMScheduler
import torch
from PIL import Image
import numpy as np


class ControlNetSegment:
    # Load pretrained model


    def __init__(self):

        # Check the CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If GPU not available in torch then throw error up instantiation
        if self.device == 'cpu':
            raise MemoryError('GPU needed for inference in this project')

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, cache_dir='D:\cache'
        )

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                        controlnet=controlnet,
                                                                        torch_dtype=torch.float16,
                                                                        safety_checker=None,
                                                                        cache_dir='D:\cache'
                                                                        )

    def make_inpaint_condition(self,image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def segment_generation(self,
                           image=None,
                           image_segm=None,
                           save_gen_path=None,
                           num_inf_steps=20,
                           scheduler=DDPMScheduler,
                           scale_num=0.6,
                           prompt=None,
                           neg_prompt=None,
                           seed=22222222):




        #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        #torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms(True)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()
        self.pipe.load_lora_weights("D:/ControlNet-v1-1-nightly/", weight_name="a_line_hairstyle.safetensors")

        control_image = self.make_inpaint_condition(image, image_segm)

        image = self.pipe(prompt,
                     negative_prompt=neg_prompt,
                     image=image,
                     mask_image=image_segm,
                     control_image=control_image,
                     generator=generator,
                     num_inference_steps=20,
                     cross_attention_kwargs={"scale": scale_num}).images[0]

        if save_gen_path is not None:
            image.save(save_gen_path)
        return image