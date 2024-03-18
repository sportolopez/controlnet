import os

import torchvision

from utils import resize_image_if_big_by_size

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import KDPM2DiscreteScheduler, StableDiffusionControlNetPipeline, ControlNetModel, \
    StableDiffusionControlNetInpaintPipeline, KarrasVeScheduler, DPMSolverMultistepScheduler, PNDMScheduler, \
    DDPMScheduler, DEISMultistepScheduler, CMStochasticIterativeScheduler, DDIMInverseScheduler, \
    UniPCMultistepScheduler, DDIMParallelScheduler, DDIMScheduler, DDPMParallelScheduler, \
    DPMSolverMultistepInverseScheduler, DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, \
    EulerDiscreteScheduler, HeunDiscreteScheduler, KDPM2AncestralDiscreteScheduler, IPNDMScheduler, \
    StableDiffusionPipeline, StableDiffusionLatentUpscalePipeline
from huggingface_hub import RepoCard
from torch import nn
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation, SegformerImageProcessor, \
    AutoModelForSemanticSegmentation, AutoModel

from controlnet_aux import HEDdetector
from compel import Compel

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')


def add_sufix_filename(ruta_completa, sufijo):
    carpeta, nombre_archivo = os.path.split(ruta_completa)
    nombre_base, extension = os.path.splitext(nombre_archivo)
    nuevo_nombre = f"{nombre_base}{sufijo}{extension}"
    nueva_ruta_completa = os.path.join(carpeta, nuevo_nombre)
    return nueva_ruta_completa


def limpiar_cara(imagen_face, image):
    # Iterar sobre los píxeles de la imagen 1
    for y in range(imagen_face.shape[0]):
        for x in range(imagen_face.shape[1]):
            if imagen_face[y, x] != 0:  # Verificar si el píxel es negro
                image[y, x] = 0  # Copiar el píxel de la imagen 1 a la imagen 2


def ensanchar_borde(imagen, dilatacion):
    # Invertir los colores (negativo)
    imagen_invertida = cv2.bitwise_not(imagen)

    # Definir el kernel para la operación de dilatación
    kernel = np.ones((dilatacion, dilatacion), np.uint8)

    # Aplicar la operación de dilatación
    borde_ensanchado = cv2.dilate(imagen, kernel, iterations=1)

    # Invertir nuevamente los colores para obtener el resultado final
    # borde_ensanchado = cv2.bitwise_not(borde_ensanchado)

    # Mostrar la imagen original y la imagen con el borde ensanchado
    # cv2.imshow("Imagen Original", imagen)
    # cv2.waitKey(0)
    # cv2.imshow("Borde Ensanchado", borde_ensanchado)
    # cv2.waitKey(0)
    return borde_ensanchado


def get_hair_segmentation(ruta_completa):
    image = Image.open(ruta_completa).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    seg_cara = upsampled_logits.argmax(dim=1)[0]
    seg_cara[seg_cara != 11] = 0
    arr_seg_cara = seg_cara.cpu().numpy().astype("uint8")
    arr_seg_cara *= 255

    seg_pelo = upsampled_logits.argmax(dim=1)[0]
    seg_pelo[seg_pelo != 2] = 0
    arr_seg = seg_pelo.cpu().numpy().astype("uint8")
    arr_seg *= 255

    image = ensanchar_borde(arr_seg, 40)
    limpiar_cara(arr_seg_cara, image)

    pil_seg = Image.fromarray(image)

    nueva_ruta_completa = add_sufix_filename(ruta_completa, "_segm")

    pil_seg.save(nueva_ruta_completa)
    pil_seg.close()

    return nueva_ruta_completa


def make_inpaint_condition(image, image_mask):
    converted = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert converted.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    converted[image_mask > 0.5] = -1.0  # set as masked pixel
    converted = np.expand_dims(converted, 0).transpose(0, 3, 1, 2)
    converted = torch.from_numpy(converted)
    return converted


class ControlNetSegment:
    def __init__(self, prompt, image_path, image_processor='shi-labs/oneformer_ade20k_swin_large',
                 image_segment="openmmlab/upernet-convnext-small",
                 pretrain_control_net="lllyasviel/control_v11p_sd15_inpaint",
                 pretrain_stable_diffusion="SG161222/Realistic_Vision_V5.1_noVAE"):
        # Set our class variables
        self.image_path = image_path
        # Get the pretrained models from HuggingFace
        # self.image_processor = AutoImageProcessor.from_pretrained(image_processor)

        # modelo del Preprocessor
        self.control_net = pretrain_control_net
        # self.control_net = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")

        self.stable_diffusion = pretrain_stable_diffusion
        self.prompt = prompt

        # Check the CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If GPU not available in torch then throw error up instantiation
        if self.device == 'cpu':
            raise MemoryError('GPU needed for inference in this project')

        # Raise error if assert statement is not met
        assert isinstance(image_path, str), 'Image path must be a string linking to an image'

    def segment_generation(self,
                           segm_image=None,
                           save_gen_path=None,
                           num_inf_steps=20,
                           scheduler=None,
                           scale_num=0.6):
        # Set variable
        image_mask = Image.open(segm_image)
        image = Image.open(self.image_path)

        # Load pretrained model
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, cache_dir='D:\cache'
        )

        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
        generator = torch.Generator(device="cuda").manual_seed(22222222)

        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE",
                                                                        controlnet=controlnet,
                                                                        torch_dtype=torch.float16,
                                                                        safety_checker=None,
                                                                        use_safetensors=True,
                                                                        cache_dir='D:\cache',
                                                                        use_auth_token="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"
                                                                        ).to("cuda")
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.load_lora_weights("C:/Users/Administrator/git/controlnet/loras/",
                               weight_name="egyptian_bob_hairstyle.safetensors")

        control_image = make_inpaint_condition(image, image_mask)
        # torchvision.utils.save_image(control_image,self.image_path + "control.jpg")
        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        # prompt_embeds = compel_proc()
        prompt = 'egyptian_bob_hairstyle, 4k, high-res, masterpiece, best quality,((Hasselblad photography)), finely detailed skin, sharp focus, (cinematic lighting), soft lighting, dynamic angle'

        low_res_latents = pipe(prompt,
                               negative_prompt="(greyscale:1.2),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans",
                               image=image,
                               mask_image=image_mask,
                               control_image=control_image,
                               generator=generator,
                               num_inference_steps=20,
                               guidance_scale=7,
                               cross_attention_kwargs={"scale": scale_num}, output_type="latent").images

        with torch.no_grad():
            image = pipe.decode_latents(low_res_latents)
        image = pipe.numpy_to_pil(image)[0]

        pipe_upscale = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler",
                                                                            torch_dtype=torch.float16)

        image = pipe_upscale(
            image=image,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
        ).images[0]

        if save_gen_path is not None:
            save_gen_path = save_gen_path.replace("jpg", "png")
            image.save(save_gen_path, quality=100)
        return image


if __name__ == '__main__':
    prompt = '(egyptian_bob_hairstyle:1.5), 4k, high-res, masterpiece, best quality,((Hasselblad photography)), finely detailed skin, sharp focus, (cinematic lighting), soft lighting, dynamic angle'
    img_path = "./images/01.jpg"

    schedulers = [
        # ("UniPCMultistepScheduler", UniPCMultistepScheduler),
        # ("CMStochasticIterativeScheduler", CMStochasticIterativeScheduler),
        # ("DDIMInverseScheduler", DDIMInverseScheduler),
        # ("DDIMParallelScheduler", DDIMParallelScheduler),
        # ("DDIMScheduler", DDIMScheduler),
        # ("DDPMParallelScheduler", DDPMParallelScheduler),
        # ("DDPMScheduler", DDPMScheduler),
        # ("DEISMultistepScheduler", DEISMultistepScheduler),
        # ("DPMSolverMultistepInverseScheduler", DPMSolverMultistepInverseScheduler),
        # ("DPMSolverMultistepScheduler", DPMSolverMultistepScheduler),
        # ("DPMSolverMultistepScheduler_karras", DPMSolverMultistepScheduler(use_karras_sigmas=True)),
        # ("DPMSolverSinglestepScheduler", DPMSolverSinglestepScheduler),
        ("EulerAncestralDiscreteScheduler", EulerAncestralDiscreteScheduler),
        # ("EulerDiscreteScheduler", EulerDiscreteScheduler),
        # ("HeunDiscreteScheduler", HeunDiscreteScheduler),
        # ("IPNDMScheduler", IPNDMScheduler),
        # ("KarrasVeScheduler", KarrasVeScheduler),
        # ("KDPM2AncestralDiscreteScheduler", KDPM2AncestralDiscreteScheduler),
        # ("KDPM2DiscreteScheduler", KDPM2DiscreteScheduler),
        # ("PNDMScheduler", PNDMScheduler)
    ]

    path = "images"
    archivos = os.listdir(path)
    archivos = [archivo for archivo in archivos if os.path.isfile(os.path.join(path, archivo))]
    archivos = [archivo for archivo in archivos if "_segm" not in archivo]
    archivos = [archivo for archivo in archivos if "_gen" not in archivo]
    archivos = [archivo for archivo in archivos if "_face" not in archivo]

    for archivo in archivos:
        print("Inicio imagen:" + archivo)
        ruta_completa = os.path.join(path, archivo)
        # resize_image_if_big_by_size(ruta_completa,512)

        control_net_seg = ControlNetSegment(
            prompt=prompt,
            image_path=ruta_completa)
        for aScheduler in schedulers:
            nombreScheduler = aScheduler[0]
            print("Inicio Sche:" + nombreScheduler)
            inicio = time.time()

            for i in range(5, 6, 1):
                varScale = i / 10.0
                nueva_ruta_completa = add_sufix_filename(ruta_completa,
                                                         "_" + nombreScheduler + "_" + str(varScale) + "_gen")
                ruta_segmentation = add_sufix_filename(ruta_completa, "_segm")
                # resize_image_if_big_by_size(ruta_segmentation, 512)
                seg_image = control_net_seg.segment_generation(
                    segm_image=ruta_segmentation,
                    save_gen_path=nueva_ruta_completa,
                    scheduler=aScheduler[1],
                    scale_num=varScale
                )
                fin = time.time()
                print(f"Tiempo de ejecución: {fin - inicio} segundos {nueva_ruta_completa}")
