import base64
import unittest
import importlib

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

utils.setup_test_env()


def limpiar_cara(imagen_face, image):
    # Iterar sobre los píxeles de la imagen 1
    for y in range(imagen_face.shape[0]):
        for x in range(imagen_face.shape[1]):
            if imagen_face[y, x] != 0:  # Verificar si el píxel es negro
                image[y, x] = 0  # Copiar el píxel de la imagen 1 a la imagen 2

    # Guardar la imagen resultante
    cv2.imwrite("imagen_limpiar_cara.png", image)


def ensanchar_borde(imagen, dilatacion):
    # Invertir los colores (negativo)
    imagen_invertida = cv2.bitwise_not(imagen)

    # Definir el kernel para la operación de dilatación
    kernel = np.ones((dilatacion, dilatacion), np.uint8)

    # Aplicar la operación de dilatación
    borde_ensanchado = cv2.dilate(imagen, kernel, iterations=1)

    # Invertir nuevamente los colores para obtener el resultado final
    #borde_ensanchado = cv2.bitwise_not(borde_ensanchado)

    # Mostrar la imagen original y la imagen con el borde ensanchado
    #cv2.imshow("Imagen Original", imagen)
    #cv2.waitKey(0)
    #cv2.imshow("Borde Ensanchado", borde_ensanchado)
    #cv2.waitKey(0)
    return borde_ensanchado


def get_hair_segmentation(image):
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

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

    image = ensanchar_borde(arr_seg, 50)
    limpiar_cara(arr_seg_cara, image)

    pil_seg = Image.fromarray(image)
    pil_seg.save("segmentation.png")
    pil_seg.close()



# get_hair_segmentation(Image.open("mujer1.PNG").convert("RGB"))


class TestAlwaysonTxt2ImgWorking(unittest.TestCase):

    get_hair_segmentation(Image.open("images\\mujer3.jpg").convert("RGB"))

    exit(1)

    def setUp(self):
        controlnet_unit = {
            "enabled": True,
            "module": "inpaint_only",
            "model": "control_v11p_sd15_inpaint [ebff9138]",
            "weight": 1.0,
            "image": utils.readImage("images\\mujer3.jpg"),
            "mask":  utils.readImage("segmentation.png"),
            "resize_mode": 1,
            "lowvram": False,
            "processor_res": 512,
            "threshold_a": -1,
            "threshold_b": -1,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "control_mode": 0,
            "pixel_perfect": False
        }
        setup_args = [controlnet_unit] * getattr(self, 'units_count', 1)
        self.setup_route(setup_args)

    def setup_route(self, setup_args):
        self.url_txt2img = "http://localhost:7860/sdapi/v1/txt2img"
        self.simple_txt2img = {
                    "enable_hr": True,
                    "denoising_strength": 1,
                    "firstphase_width": 0,
                    "firstphase_height": 0,
                    "prompt": "(hi_top_fade_hairstyle:1.3),woman posing for a photo, good hand,4k, high-res, masterpiece, best quality, head:1.3,((Hasselblad photography)), finely detailed skin, sharp focus, (cinematic lighting), soft lighting, dynamic angle, [:(detailed face:1.2):0.2],  <lora:hi_top_fade_hairstyle:0.5> ",
                    "negative_prompt": "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, skin spots, acnes, skin blemishes, age spot, glans",
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
                    "width": 512,
                    "height": 512,
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
                    "hr_scale": 1,
                    "hr_upscaler": "None",
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
        self.setup_controlnet_params(setup_args)

    def setup_controlnet_params(self, setup_args):
        self.simple_txt2img["alwayson_scripts"]["ControlNet"] = {
            "args": setup_args
        }

    def assert_status_ok(self, msg=None):
        print(self.simple_txt2img)
        response = requests.post(self.url_txt2img, json=self.simple_txt2img)
        self.assertEqual(response.status_code, 200, msg)
        #print(response.json()['images'])
        for index,aImage in enumerate(response.json()['images']):
            decoded_data = base64.b64decode(aImage)
            img_file = open('image_'+str(index)+'.png', 'wb')
            img_file.write(decoded_data)
            img_file.close()


        stderr = ""
        with open('C:/Users/sebap/git/controlnet/stderr.txt') as f:
            stderr = f.read().lower()
        with open('C:/Users/sebap/git/controlnet/stderr.txt', 'w') as f:
            # clear stderr file so that we can easily parse the next test
            f.write("")
        self.assertFalse('error' in stderr, "Errors in stderr: \n" + stderr)

    def test_txt2img_simple_performed(self):
        self.assert_status_ok()


if __name__ == "__main__":
    unittest.main()
