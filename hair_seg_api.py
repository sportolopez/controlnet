import base64
import socket
import time
import unittest
import importlib
from http.client import HTTPConnection

import cv2
import numpy as np
import torch.nn as nn
import sys
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import os
import sys
from base64 import b64encode

import utils
import requests

PATH = "images"

utils.setup_test_env()

host = "https://87a0419dd81c45550a.gradio.live/"
#host = "http://127.0.0.1:7860/"
url_txt2img = host + "sdapi/v1/txt2img"
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

processorFace = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing",resume_download=True)
modelFace = AutoModelForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing",resume_download=True)


def limpiar_cara(imagen_face, image):
    # Iterar sobre los píxeles de la imagen 1
    for y in range(imagen_face.shape[0]):
        for x in range(imagen_face.shape[1]):
            if imagen_face[y, x] != 0:  # Verificar si el píxel es negro
                image[y, x] = 0  # Copiar el píxel de la imagen 1 a la imagen 2



def ensanchar_borde(imagen, dilatacion):
    # Definir el kernel para la operación de dilatación
    kernel = np.ones((dilatacion, dilatacion), np.uint8)

    # Aplicar la operación de dilatación
    borde_ensanchado = cv2.dilate(imagen, kernel, iterations=1)

    return borde_ensanchado

def get_face_segmentation(ruta_completa):
    image = Image.open(ruta_completa).convert("RGB")
    inputs = processorFace(images=image, return_tensors="pt")
    outputs = modelFace(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    seg_pelo = upsampled_logits.argmax(dim=1)[0]
    #seg_pelo[seg_pelo != 5] = 0
    mask = (seg_pelo != 1) & (seg_pelo != 2) & (seg_pelo != 3) & (seg_pelo != 4) & (seg_pelo != 5) & (seg_pelo != 6) & (seg_pelo != 7) & (seg_pelo != 9) & (seg_pelo != 10) & (seg_pelo != 11) & (seg_pelo != 12) &(seg_pelo != 16)
    seg_pelo[mask] = 0
    arr_seg = seg_pelo.cpu().numpy().astype("uint8")
    arr_seg *= 255

    #no se por que algunos byte no estan en 255
    alto, ancho = arr_seg.shape
    for y in range(alto - 1, -1, -1):
        for x in range(ancho):
            if arr_seg[y, x] != 0:
                arr_seg[y, x] = 255

    arr_seg = cv2.bitwise_not(arr_seg)
    imagen_ceja_i = get_image_by_byte(upsampled_logits.argmax(dim=1)[0],7)#uso el id de la oreja por que siempre lo identifica aca
    imagen_ceja_d = get_image_by_byte(upsampled_logits.argmax(dim=1)[0],6)
    imagen_labio_inf = get_image_by_byte(upsampled_logits.argmax(dim=1)[0],12)#uso la del cuello por que no identifica bien labio inf
    lower_point = get_lower_point(imagen_labio_inf)
    image_ensanchada = ensanchar_borde2(arr_seg, 150)
    image_clean = get_a_line_haircut(arr_seg,image_ensanchada,imagen_ceja_d,imagen_ceja_i,lower_point)


    nueva_ruta_completa = add_sufix_filename(ruta_completa, "_face")
    pil_seg = Image.fromarray(image_ensanchada)
    pil_seg.save(nueva_ruta_completa)
    pil_seg.close()

def get_image_by_byte(img_array,byte_id):
    img_array[img_array != byte_id] = 0
    img_array[img_array != 0] = 255
    arr_seg = img_array.cpu().numpy().astype("uint8")
    imagen = cv2.bitwise_not(arr_seg)
    return imagen

def get_lower_point(imagen):
    # Obtener las dimensiones de la imagen
    alto, ancho = imagen.shape

    # Inicializar las coordenadas del punto más bajo
    x_masbajo = -1
    y_masbajo = -1

    # Iterar sobre las filas desde abajo hacia arriba
    for y in range(alto - 1, -1, -1):
        for x in range(ancho):
            if imagen[y, x] != 255:
                print(imagen[y, x])
            if imagen[y, x] == 0:  # Si el píxel es negro (0 en escala de grises)
                x_masbajo = x
                y_masbajo = y
                break
        if x_masbajo != -1:
            break

    # Verificar si se encontró un píxel negro (forma irregular presente)
    if x_masbajo != -1:
        print(f"El punto más bajo está en las coordenadas: ({x_masbajo}, {y_masbajo})")
    else:
        print("No se encontró una forma irregular en la imagen.")
    return x_masbajo, y_masbajo

def get_a_line_haircut(imagen_face, image, image_ceja_r, image_ceja_l, lower_point):
    coordenadas = cv2.minMaxLoc(image_ceja_r)
    x, y_ceja_r = coordenadas[2]
    coordenadas = cv2.minMaxLoc(image_ceja_l)
    x, y_ceja_l = coordenadas[2]
    y_ceja = y_ceja_r if y_ceja_r > y_ceja_l else y_ceja_l

    # Iterar sobre los píxeles de la imagen 1
    for y in range(imagen_face.shape[0]):
        for x in range(imagen_face.shape[1]):
            if (imagen_face[y, x] == 0 and #Elimina la cara
                    # Agrega la frente
                    y > y_ceja - 10) or \
                    y > lower_point[0]: #Corte hasta un Y determinado (obtenido por el borde de la boca)
                image[y, x] = 255  #
    return image

def ensanchar_borde2(imagen, dilatacion):
    # Invertir los colores (negativo)
    imagen_invertida = cv2.bitwise_not(imagen)

    # Definir el kernel para la operación de dilatación
    kernel = np.ones((dilatacion, dilatacion), np.uint8)

    # Aplicar la operación de dilatación
    borde_ensanchado = cv2.dilate(imagen_invertida, kernel, iterations=1)

    # Invertir nuevamente los colores para obtener el resultado final
    borde_ensanchado = cv2.bitwise_not(borde_ensanchado)

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


def add_sufix_filename(ruta_completa, sufijo):
    carpeta, nombre_archivo = os.path.split(ruta_completa)
    nombre_base, extension = os.path.splitext(nombre_archivo)
    nuevo_nombre = f"{nombre_base}{sufijo}{extension}"
    nueva_ruta_completa = os.path.join(carpeta, nuevo_nombre)
    return nueva_ruta_completa


class TestAlwaysonTxt2ImgWorking(unittest.TestCase):

    def setUpControlnet(self, image_path, seg_path):
        read_image_original, wid, hgt = utils.readImage(image_path)
        print("Resolution:"+str(wid) + " "+str(hgt))
        read_image_seg, seg_wid, seg_hgt = utils.readImage(seg_path)
        resolution = [wid,hgt]
        controlnet_unit = {
            "enabled": True,
            "module": "inpaint_only",
            "model": "control_v11p_sd15_inpaint [ebff9138]",
            "weight": 1.0,
            "image": read_image_original,
            "mask":  read_image_seg,
            "resize_mode": 1,
            "lowvram": False,
            #"processor_res": resolution[0],
            "threshold_a": -1,
            "threshold_b": -1,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "control_mode": 0,
            "pixel_perfect": False
        }
        setup_args = [controlnet_unit] * getattr(self, 'units_count', 1)
        prompt = "(a_line_haircut:1.3), 4k, high-res, masterpiece, best quality,((Hasselblad photography)), finely detailed skin, sharp focus, (cinematic lighting), soft lighting, dynamic angle,  <lora:a_line_hairstyle:0.5> "
        return self.setup_route(setup_args,resolution, prompt)

    def setup_route(self, setup_args,resolution, prompt):
        simple_txt2img = {
                    "enable_hr": True,
                    #"denoising_strength": 1,
                    "firstphase_width": 0,
                    "firstphase_height": 0,
                    "prompt": prompt,
                    "negative_prompt": "(greyscale:1.2),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans",
                    "styles": [],
                    "seed": 22222222,
                    "subseed": -1,
                    "subseed_strength": 0,
                    "seed_resize_from_h": -1,
                    "seed_resize_from_w": -1,
                    "sampler_index": "DPM++ 2M Karras",
                    "batch_size": 1,
                    "n_iter": 1,
                    "steps": 20,
                    "cfg_scale": 7,
                    "width": resolution[0],
                    "height": resolution[1],
                    "restore_faces": False,
                    "tiling": False,
                    "eta": 0,
                    "s_churn": 0,
                    "s_tmax": 0,
                    "s_tmin": 0,
                    "s_noise": 0,
                    "s_min_uncond": 0,
                    "override_settings": {},
                    "override_settings_restore_afterwards": True,
                    "refiner_checkpoint": "",
                    "refiner_switch_at": 0,
                    "disable_extra_networks": False,
                    "comments": {},
                    #"hr_scale": 1,
                    #"hr_upscaler": "None",
                    "hr_second_pass_steps": 0,
                    "hr_resize_x": 0,
                    "hr_resize_y": 0,
                    "hr_checkpoint_name": "",
                    #"hr_sampler_name": "",
                    "hr_prompt": "",
                    "hr_negative_prompt": "",
                    "script_name": "",
                    "script_args": [],
                    "send_images": True,
                    "save_images": False,
                    "do_not_save_samples": False,
                    "do_not_save_grid": False,
                    "sampler_name": "",
                    "alwayson_scripts": {}
        }
        simple_txt2img["alwayson_scripts"]["ControlNet"] = {
            "args": setup_args
        }
        return simple_txt2img

    def assert_status_ok(self, msg=None):
        archivos = os.listdir(PATH)
        archivos = [archivo for archivo in archivos if os.path.isfile(os.path.join(PATH, archivo))]
        archivos = [archivo for archivo in archivos if "_segm" not in archivo]
        archivos = [archivo for archivo in archivos if "_gen" not in archivo]
        archivos = [archivo for archivo in archivos if "_face" not in archivo]
        # Imprime la lista de archivos
        for archivo in archivos:
            print("Inicio imagen:" + archivo)
            ruta_completa = os.path.join(PATH, archivo)
            utils.resize_image_if_big(ruta_completa)
            inicio = time.time()
            nueva_ruta_completa = get_hair_segmentation(ruta_completa)
            get_face_segmentation(ruta_completa)
            fin = time.time()
            print(f"Tiempo de ejecución: {fin - inicio} segundos")
            json_body = self.setUpControlnet(image_path=ruta_completa, seg_path=nueva_ruta_completa)
            '''
            print("Inicio post")
            inicio = time.time()
            response = requests.post(url=url_txt2img, json=json_body)
            if(response.status_code != 200):
                print(f"Error {response.status_code}")
                print(f"Error {url_txt2img}")
                print(f"Error {response.headers}")
                print(f"Error {response.body}")
                exit(1)
            fin = time.time()
            print(f"Tiempo de ejecución: {fin - inicio} segundos")
            decoded_data = base64.b64decode(response.json()['images'][0])
            img_file = open(add_sufix_filename(ruta_completa, "_gen"), 'wb')
            img_file.write(decoded_data)
            img_file.close()
            

        stderr = ""
        with open('stderr.txt', 'w') as f:
            stderr = f.read().lower()
        with open('stderr.txt', 'w') as f:
            # clear stderr file so that we can easily parse the next test
            f.write("")
        self.assertFalse('error' in stderr, "Errors in stderr: \n" + stderr)'''

    def test_txt2img_simple_performed(self):
        self.assert_status_ok()


if __name__ == "__main__":
    unittest.main()
