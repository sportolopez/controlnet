import os
import sys
import cv2
from base64 import b64encode

import requests

BASE_URL = "http://localhost:7860"


def setup_test_env():
    ext_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if ext_root not in sys.path:
        sys.path.append(ext_root)


def readImage(path):
    img = cv2.imread(path)
    # fetching the dimensions
    wid = img.shape[1]
    hgt = img.shape[0]
    retval, buffer = cv2.imencode('.jpg', img)
    b64img = b64encode(buffer).decode("utf-8")
    return b64img, wid, hgt


def resize_image_if_big_by_size(image_path,max_w_h):
    # Lee la imagen
    img = cv2.imread(image_path)

    # Obtiene el ancho y el alto
    height, width, _ = img.shape

    # Verifica si el ancho o el alto superan los 1024
    if width > max_w_h or height > max_w_h:
        print("Se redimensiona la imagen a "+str(max_w_h))
        # Redimensiona la imagen manteniendo la relación de aspecto
        if width >= height:
            new_width = max_w_h
            new_height = int(max_w_h * height / width)
        else:
            new_height = max_w_h
            new_width = int(max_w_h * width / height)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Sobreescribe la imagen original
        cv2.imwrite(image_path, resized_img)

def resize_image_if_big(image_path):
    # Lee la imagen
    resize_image_if_big_by_size(image_path,1024)




def get_model():
    r = requests.get(BASE_URL+"/controlnet/model_list")
    result = r.json()
    if "model_list" in result:
        result = result["model_list"]
        for item in result:
            print("Using model: ", item)
            return item
    return "None"


def get_modules():
    return requests.get(f"{BASE_URL}/controlnet/module_list").json()


def detect(json):
    return requests.post(BASE_URL+"/controlnet/detect", json=json)
