import time

import cv2
import numpy as np
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import os

import datetime

PATH = "images"

#utils.setup_test_env()

#host = "https://87a0419dd81c45550a.gradio.live/"
host = "http://127.0.0.1:7860/"
url_txt2img = host + "sdapi/v1/txt2img"
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

processorFace = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing", resume_download=True)
modelFace = AutoModelForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing", resume_download=True)


def hora():
    hora_inicio = datetime.datetime.now()
    hora_inicio_formateada = hora_inicio.strftime('%H:%M:%S.%f')[:-3]
    print(f'Hora: {hora_inicio_formateada}')


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


def get_face_segmentation(image, pelo_largo):
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
    # seg_pelo[seg_pelo != 5] = 0
    mask = (seg_pelo != 1) & (seg_pelo != 2) & (seg_pelo != 4) & (seg_pelo != 5) & (seg_pelo != 6) & (seg_pelo != 7) & (
                seg_pelo != 10) & (seg_pelo != 11) & (seg_pelo != 12)

    seg_pelo[mask] = 0
    arr_seg = seg_pelo.cpu().numpy().astype("uint8")
    # no se por que algunos byte no estan en 255
    pixeles_no_cero = arr_seg != 0
    arr_seg[pixeles_no_cero] = 255
    arr_seg = cv2.bitwise_not(arr_seg)
    imagen_ceja_i = get_image_by_byte(upsampled_logits.argmax(dim=1)[0],
                                      7)  # uso el id de la oreja por que siempre lo identifica aca

    imagen_ceja_d = get_image_by_byte(upsampled_logits.argmax(dim=1)[0], 6)
    imagen_labio_inf = get_image_by_byte(upsampled_logits.argmax(dim=1)[0],
                                         12)  # uso la del cuello por que no identifica bien labio inf

    lower_point = get_lower_point(imagen_labio_inf)

    imagen_cuello = ''
    imagen_ropa = ''

    if pelo_largo:
        image_ensanchada = ensanchar_borde2(arr_seg, 175)
        imagen_cuello = get_image_by_byte(upsampled_logits.argmax(dim=1)[0],17)
        imagen_ropa = get_image_by_byte(upsampled_logits.argmax(dim=1)[0], 18)
    else:
        image_ensanchada = ensanchar_borde2(arr_seg, 150)

    image_clean = get_a_line_haircut(arr_seg, image_ensanchada, imagen_ceja_d, imagen_ceja_i, lower_point, pelo_largo)
    '''
    nueva_ruta_completa = add_sufix_filename(ruta_completa, "_face")

    pil_seg = Image.fromarray(image_clean)
    pil_seg.save(nueva_ruta_completa)
    pil_seg.close()'''
    return image_clean, imagen_cuello, imagen_ropa, arr_seg


def get_image_by_byte(img_array, byte_id):
    img_array[img_array != byte_id] = 0
    img_array[img_array != 0] = 255
    arr_seg = img_array.cpu().numpy().astype("uint8")
    imagen = cv2.bitwise_not(arr_seg)
    return imagen


def get_lower_point(imagen):
    # Encontrar el índice del primer píxel no blanco
    indice_no_blanco = np.where(imagen != 255)

    # Si no se encontraron píxeles no blancos
    if len(indice_no_blanco[0]) == 0:
        x_masbajo = -1
        y_masbajo = -1
    else:
        # Obtener el último píxel no blanco
        y_masbajo = indice_no_blanco[0][-1]
        x_masbajo = indice_no_blanco[1][-1]

    print(f"El punto más bajo está en las coordenadas: ({x_masbajo}, {y_masbajo})")
    return x_masbajo, y_masbajo


def get_a_line_haircut(imagen_face, image, image_ceja_r, image_ceja_l, lower_point, pelo_largo):
    coordenadas = cv2.minMaxLoc(image_ceja_r)
    x, y_ceja_r = coordenadas[2]
    coordenadas = cv2.minMaxLoc(image_ceja_l)
    x, y_ceja_l = coordenadas[2]
    y_ceja = y_ceja_l if y_ceja_r == 0 else y_ceja_r if y_ceja_l == 0 else min(y_ceja_r, y_ceja_l)

    altura_imagen = imagen_face.shape[0]
    # Crear una máscara para los píxeles que deben mantenerse
    limite_y = y_ceja - 10
    imagen_face[0:limite_y, :] = 255
    if not pelo_largo:
        image[lower_point[1]:altura_imagen, :] = 255
    mascara_pepito = (imagen_face != 0)
    mascara_pepito2 = (image == 0)
    image = np.where(mascara_pepito & mascara_pepito2, 0, 255)
    image = image.astype(np.uint8)
    return image


def ensanchar_borde2(imagen, dilatacion):
    # Invertir los colores (negativo)
    imagen_invertida = cv2.bitwise_not(imagen)

    # Definir el kernel para la operación de dilatación
    # KERNEL CUADRADO
    # kernel = np.ones((dilatacion, dilatacion), np.uint8)
    # borde_ensanchado = cv2.dilate(imagen_invertida, kernel, iterations=1)
    # borde_ensanchado = cv2.bitwise_not(borde_ensanchado)

    # KERNEL CIRCULAR
    kernel_circular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatacion, dilatacion))
    borde_ensanchado_circular = cv2.dilate(imagen_invertida, kernel_circular, iterations=1)
    borde_ensanchado_circular = cv2.bitwise_not(borde_ensanchado_circular)

    return borde_ensanchado_circular


def get_hair_segmentation(image):
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
    '''
    pil_seg = Image.fromarray(image)

    nueva_ruta_completa = add_sufix_filename(ruta_completa, "_segm")

    pil_seg.save(nueva_ruta_completa)
    pil_seg.close()
    '''
    return image


def extender_mascara(imagen_unida, imagen_ropa, imagen_cuello, face):
    coordenadas = np.argwhere(imagen_unida == 255)

    objeto_y_mas_alto = max(coordenadas, key=lambda item: item[1])
    objeto_y_mas_bajo = min(coordenadas, key=lambda item: item[1])

    # Crear una máscara para los píxeles que deben mantenerse
    imagen_unida[objeto_y_mas_alto[0]: imagen_unida.shape[0], objeto_y_mas_bajo[1]:objeto_y_mas_alto[1]] = 255

    # image[lower_point[1]:altura_imagen, :] = 255
    mascara_pepito = (imagen_unida != 0)
    mascara_face = (face == 255)
    image = np.where(mascara_pepito & imagen_ropa & imagen_cuello & mascara_face, 255, 0)
    image = image.astype(np.uint8)
    return image


def add_sufix_filename(ruta_completa, sufijo):
    carpeta, nombre_archivo = os.path.split(ruta_completa)
    nombre_base, extension = os.path.splitext(nombre_archivo)
    nuevo_nombre = f"{nombre_base}{sufijo}{extension}"
    nueva_ruta_completa = os.path.join(carpeta, nuevo_nombre)
    return nueva_ruta_completa


def segment_hair(image, pelo_largo=False):
    image = image.convert("RGB")

    #utils.resize_image_if_big(ruta_completa)
    inicio = time.time()
    image_hair = get_hair_segmentation(image)
    image_face, imagen_cuello, imagen_ropa, face = get_face_segmentation(image, pelo_largo)
    image_hair = cv2.bitwise_not(image_hair)
    imagen_unida = cv2.bitwise_and(image_hair, image_face)
    imagen_unida = cv2.bitwise_not(imagen_unida)
    if pelo_largo:
        imagen_unida = extender_mascara(imagen_unida, imagen_ropa, imagen_cuello, face)
    tiempo_transcurrido = time.time() - inicio
    print(f"******La ejecución de segment_hair tardó {tiempo_transcurrido} segundos")
    return Image.fromarray(imagen_unida)





if __name__ == "__main__":

    segmentacion = segment_hair(Image.open("../../images/20.png"), True)
    segmentacion.save("../../images/20_segamano_largo.jpg")
    segmentacion.close()
