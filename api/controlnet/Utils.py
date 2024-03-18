import cv2
import numpy as np
from PIL import Image


def resize_image_if_big_by_size(image,max_w_h):
    # Lee la imagen
    img = np.array(image)

    # Obtiene el ancho y el alto
    if len(img.shape) == 2:
        # Grayscale image
        height, width = img.shape
        channels = 1
    else:
        # Color image
        height, width, channels = img.shape

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
        resized_pil_image = Image.fromarray(resized_img)

        return resized_pil_image
        # Sobreescribe la imagen original
    else:
        return image
