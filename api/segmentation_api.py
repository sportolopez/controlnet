import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
    EulerDiscreteScheduler, HeunDiscreteScheduler, KDPM2AncestralDiscreteScheduler, IPNDMScheduler
from huggingface_hub import RepoCard
from torch import nn
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation, SegformerImageProcessor, \
    AutoModelForSemanticSegmentation, AutoModel

from controlnet_aux import HEDdetector

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




if __name__ == '__main__':
    prompt = 'a_line_haircut, 4k, high-res, masterpiece, best quality,((Hasselblad photography)), finely detailed skin, sharp focus, (cinematic lighting), soft lighting, dynamic angle,  <lora:a_line_hairstyle:0.5>'
    img_path = "./images/01.jpg"
    '''
    schedulers = [DPMSolverMultistepScheduler(use_karras_sigmas=True)]
    '''
    schedulers = [
                  #UniPCMultistepScheduler,
                  #CMStochasticIterativeScheduler,
                  #DDIMInverseScheduler,
                  #DDIMParallelScheduler,
                  #DDIMScheduler,
                  #DDPMParallelScheduler,
                  DDPMScheduler,
                  #DEISMultistepScheduler,
                  #DPMSolverMultistepInverseScheduler,
                  #DPMSolverMultistepScheduler,
                  #DPMSolverSinglestepScheduler,
                  #EulerAncestralDiscreteScheduler,
                  #EulerDiscreteScheduler,
                  #HeunDiscreteScheduler,
                  #IPNDMScheduler,
                  #KarrasVeScheduler,
                  #KDPM2AncestralDiscreteScheduler,
                  #KDPM2DiscreteScheduler,
                  #PNDMScheduler
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


        control_net_seg = ControlNetSegment(
            prompt=prompt,
            image_path=ruta_completa)
        for aScheduler in schedulers:
            nombreScheduler = aScheduler.__name__
            print("Inicio Sche:" + nombreScheduler)
            inicio = time.time()

            for i in range(5,6,1):
                varScale = i / 10.0
                nueva_ruta_completa = add_sufix_filename(ruta_completa, "_"+nombreScheduler+"_"+str(varScale)+ "_gen")
                ruta_segmentation = add_sufix_filename(ruta_completa, "_segm")
                seg_image = control_net_seg.segment_generation(
                    segm_image=ruta_segmentation,
                    save_gen_path=nueva_ruta_completa,
                    scheduler=aScheduler,
                    scale_num=varScale
                )
                fin = time.time()
                print(f"Tiempo de ejecución: {fin - inicio} segundos { nueva_ruta_completa }")

