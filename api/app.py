import base64
import time
from io import BytesIO

import connexion
import cv2
import numpy as np
from PIL import Image
from flask import Flask, send_file
import hair_seg_api
from api.ControlNetSegment import ControlNetSegment


def get_image_from_base64(base64string):
    imagen_bytes = base64.b64decode(base64string)
    image_buffer = BytesIO(imagen_bytes)
    return Image.open(image_buffer)


def generar():
    body = connexion.request.get_json()
    prompt = body['prompt']
    neg_prompt = body['neg_prompt']
    seed = (body['seed'])
    image = get_image_from_base64(body['image'])
    imagen_mask = get_image_from_base64(body['imagen_mask'])

    inicio = time.time()
    controlnet = ControlNetSegment()
    imagen_gen = controlnet.segment_generation(image=image, image_segm=imagen_mask, prompt=prompt,
                                               neg_prompt=neg_prompt, seed=seed)

    image_bytes = BytesIO()
    imagen_gen.save(image_bytes, format="PNG")  # You can choose a different format if needed
    image_bytes.seek(0)

    # En este ejemplo, simplemente devolveré un mensaje de éxito junto con los parámetros.
    return send_file(image_bytes, mimetype='image/png', as_attachment=True,
                     download_name='generated_image.png')


def status():
    # Aquí debes implementar la lógica para procesar los parámetros y generar la imagen en base64.
    # Agrega tu lógica de generación de imágenes aquí y devuelve la imagen generada en base64.

    # En este ejemplo, simplemente devolveré un mensaje de éxito junto con los parámetros.
    return {'message': 'Prueba'}


if __name__ == '__main__':
    app = connexion.App(__name__, specification_dir='./')
    app.add_api('swagger.yaml', arguments={'title': 'API de Ejemplo'})
    app.run(port=8081)
