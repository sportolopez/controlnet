import os
import time

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


        print("Antes de StableDiffusionControlNetInpaintPipeline.from_pretrained ")

        print("despues de StableDiffusionControlNetInpaintPipeline.from_pretrained ")

    def make_inpaint_condition(self,image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def read_metadata_from_safetensors(filename):
        import json

        with open(filename, mode="rb") as file:
            metadata_len = file.read(8)
            metadata_len = int.from_bytes(metadata_len, "little")
            json_start = file.read(2)

            assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
            json_data = json_start + file.read(metadata_len - 2)
            json_obj = json.loads(json_data)

            res = {}
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == '{':
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass

            return res

    def segment_generation(self,
                           image=None,
                           image_segm=None,
                           save_gen_path=None,
                           num_inf_steps=20,
                           scheduler=DDPMScheduler,
                           scale_num=0.5,
                           prompt=None,
                           neg_prompt=None,
                           lora=None,
                           seed=None):

        inicio = time.time()
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE",
                                                                        controlnet=ControlNetModel.from_pretrained(
                                                                            "lllyasviel/control_v11p_sd15_inpaint",
                                                                            torch_dtype=torch.float16,
                                                                            cache_dir='D:\cache'
                                                                        ),
                                                                        torch_dtype=torch.float16,
                                                                        safety_checker=None,
                                                                        cache_dir='D:\cache'
                                                                        )

        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = torch.Generator(device="cuda")
        self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()
        if(lora):
            self.pipe.load_lora_weights("C:/Users/Administrator/git/controlnet/loras/", weight_name=lora)
        control_image = self.make_inpaint_condition(image, image_segm)
        tiempo_transcurrido = time.time() - inicio
        print(f"****La ejecución de  from_pretrained  tardó {tiempo_transcurrido} segundos")
        inicio = time.time()
        image = self.pipe(prompt,
                     negative_prompt=neg_prompt,
                     image=image,
                     mask_image=image_segm,
                     control_image=control_image,
                     generator=generator,
                     num_inference_steps=20,
                     cross_attention_kwargs={"scale": scale_num}).images[0]

        tiempo_transcurrido = time.time() - inicio
        print(f"****La ejecución de  self.pipe  tardó {tiempo_transcurrido} segundos")

        if save_gen_path is not None:
            image.save(save_gen_path)
        return image