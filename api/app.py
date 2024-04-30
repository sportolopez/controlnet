import base64
import hashlib
import json
import logging
import re
import time
import traceback
from io import BytesIO
import jwt
import requests
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from controlnet.HairSegmentation import segment_hair
from controlnet.ControlNetSegment import ControlNetSegment
from controlnet.Loras import Loras
import connexion

controlnet = ControlNetSegment()
hashes_imagenes = {}


def get_image_from_base64(base64string):
    imagen_bytes = base64.b64decode(base64string)
    image_buffer = BytesIO(imagen_bytes)
    image = Image.open(image_buffer)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def obtener_hash_imagen(imagen):
    with BytesIO() as byte_io:
        imagen.save(byte_io, format='PNG')
        imagen_bytes = byte_io.getvalue()
    return hashlib.sha256(imagen_bytes).hexdigest()


def get_segmentacion(imagen, bool_pelo_largo):

    # Obtiene el hash de la imagen
    inicio_total = time.time()
    hash_imagen = obtener_hash_imagen(imagen)
    print(f"****obtener_hash_imagen tardó {time.time() - inicio_total} segundos")

    # Verifica si el hash de la imagen ya ha sido procesado previamente

    if hash_imagen in hashes_imagenes:
        return hashes_imagenes[hash_imagen]

    # Si el hash de la imagen no ha sido procesado previamente, procesa la imagen

    imagen_segmentada = segment_hair(imagen, bool_pelo_largo)

    # Guarda el hash de la imagen junto con la imagen segmentada
    hashes_imagenes[hash_imagen] = imagen_segmentada

    return imagen_segmentada

segmentaciones = {}
def segmentar():
    body = connexion.request.get_json()
    imagen_base64 = body['image']
    if(imagen_base64.startswith('data:image/')):
        image = get_image_from_base64(body['image'].split(",")[1])
    else:
        image = get_image_from_base64(body['image'])

    hash_imagen = obtener_hash_imagen(image)

    mask_largo = get_segmentacion(image, True)
    mask_corto = get_segmentacion(image, False)

    segmentaciones[hash_imagen] = (mask_largo, mask_corto)

    return hash_imagen


def generar():
    '''
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        print("No se encontró el encabezado de autorización.")
        return jsonify({'error': 'No se encontró el encabezado de autorización.'}), 401

    token_parts = auth_header.split()
    if len(token_parts) != 2 or token_parts[0].lower() != 'bearer':
        print("Encabezado de autorización inválido.")
        return jsonify({'error': 'Encabezado de autorización inválido.'}), 401

    token_jwt = token_parts[1]

    # Decodificar el JWT
    try:
        payload = jwt.decode(token_jwt, 'secret_key', algorithms=['HS256'])
        print("Token válido:", payload)
    except jwt.ExpiredSignatureError:
        print("Token expirado.")
        return jsonify({'error': 'Token expirado'}), 401
    except jwt.InvalidTokenError:
        print("Token inválido.")
        return jsonify({'error': 'Token inválido'}), 401
'''

    ''' set default'''
    prompt_default = '4k, high-res, masterpiece, best quality,((Hasselblad photography)), finely detailed skin, sharp focus, (cinematic lighting), soft lighting, dynamic angle'
    neg_prompt_default = '(greyscale:1.2),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans'
    key_word = ''

    inicio_total = time.time()
    body = connexion.request.get_json()

    if 'lora' in body and body['lora']:
        lora = (body['lora'])
    else:
        lora = None

    bool_pelo_largo = False
    if lora:
        bool_pelo_largo = Loras.get(lora, "")
        key_word = re.sub(r'[^a-zA-Z_]', '', lora.split('.')[0])



    imagen_base64 = body['image']
    if(imagen_base64.startswith('data:image/')):
        image = get_image_from_base64(body['image'].split(",")[1])
    else:
        image = get_image_from_base64(body['image'])

    if 'imagen_mask' in body and body['imagen_mask']:
        if body['imagen_mask'].startswith('data:image/'):
            imagen_mask = body['imagen_mask'].split(",")[1]
        else:
            imagen_mask = body['imagen_mask']

        imagen_bytes = base64.b64decode(imagen_mask)
        image_buffer = BytesIO(imagen_bytes)
        imagen_mask = Image.open(image_buffer)
        imagen_mask = imagen_mask.convert('L')
        imagen_mask.save('imagen_mask.png')
    else:
        imagen_mask = get_segmentacion(image, bool_pelo_largo)


    if 'color' in body and body['color']:
        color = (body['color'])
    else:
        color = ""

    if 'max_size' in body and body['max_size']:
        max_size = (body['max_size'])
    else:
        max_size = 2048

    if 'seed' in body and body['seed']:
        seed = (body['seed'])
    else:
        seed = None

    if 'prompt' in body and body['prompt']:
        prompt = body['prompt']
    else:
        prompt = prompt_default

    if 'neg_prompt' in body and body['neg_prompt']:
        neg_prompt = body['neg_prompt']
    else:
        neg_prompt = neg_prompt_default

    prompt = color + ", " + prompt + " , " + key_word

    print("*****Realizando prueba******")
    print("prompt:" + prompt)
    if lora:
        print("Lora: " + lora)
    print("Neg_prompt:" + neg_prompt)
    print("max_size:" + str(max_size))
    print("Seed:" + str(seed))

    # image = resize_image_if_big_by_size(image,max_size)
    # imagen_mask = resize_image_if_big_by_size(imagen_mask, max_size)
    imagen_gen = controlnet.segment_generation(image=image, image_segm=imagen_mask, prompt=prompt,
                                               neg_prompt=neg_prompt, seed=seed, lora=lora)

    image_bytes = BytesIO()
    imagen_gen.save(image_bytes, format="PNG")  # You can choose a different format if needed
    image_bytes.seek(0)
    imagen_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
    print(f"****La ejecución de  TOTAL  tardó {time.time() - inicio_total} segundos")
    # En este ejemplo, simplemente devolveré un mensaje de éxito junto con los parámetros.
    return "data:image/png;base64," + imagen_base64


def status():
    # Aquí debes implementar la lógica para procesar los parámetros y generar la imagen en base64.
    # Agrega tu lógica de generación de imágenes aquí y devuelve la imagen generada en base64.
    # foto
    # lora
    # color
    # 
    # En este ejemplo, simplemente devolveré un mensaje de éxito junto con los parámetros.
    try:
        inicio_total = time.time()

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        prompt = "a photo of an astronaut riding a horse on mars"
        image = pipe(prompt).images[0]
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")  # You can choose a different format if needed
        image_bytes.seek(0)
        print(f"****La ejecución de  TOTAL  tardó {time.time() - inicio_total} segundos")
        return send_file(image_bytes, mimetype='image/png', as_attachment=True,
                         download_name='generated_image.png')
    except Exception as e:
        # Registrar el error
        # Devolver un mensaje de error genérico al navegador
        return "Ha ocurrido un error al procesar la imagen", 500


def validate_access_token(access_token):
    # Construir la URL para obtener la información del usuario
    url = 'https://www.googleapis.com/oauth2/v3/userinfo'

    # Construir el encabezado con el token de acceso
    headers = {
        'Authorization': 'Bearer ' + access_token
    }

    # Realizar la solicitud GET para obtener la información del usuario
    response = requests.get(url, headers=headers)

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        # La solicitud fue exitosa, devolver la información del usuario
        user_info = response.json()
        return user_info
    else:
        # La solicitud no fue exitosa, imprimir el mensaje de error
        print('Error al validar el token:', response.text)
        return None


def login():
    access_token = request.args.get('access_token')
    if not access_token:
        return jsonify({'error': 'Token missing'}), 400

    try:
        token_info = validate_access_token(access_token)
        if (token_info is None):
            return jsonify({'error': "Access token invalido"}), 401
        print(token_info)
        # Generar JWT
        jwt_token = jwt.encode({'token_info': token_info}, 'secret_key', algorithm='HS256')

        return jsonify({'jwt_token': jwt_token, 'user_info': token_info}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 401


def custom_error_handler(exception):
    logging.basicConfig(level=logging.ERROR)

    # Get the exception type and message
    error_type = str(type(exception).__name__)
    error_message = str(exception)

    # Get the stack trace
    stack_trace = traceback.format_exc()

    logging.error(f"Error: {error_type} - {error_message}\n{stack_trace}")

    # Create a response JSON
    response = {
        "error": "NotFound",
        "detail": error_message,
        "stack_trace": stack_trace
    }
    return json.dumps(response), 500


my_app = connexion.FlaskApp(__name__, specification_dir='./')
my_app.add_api('swagger.yaml', arguments={'title': 'API de Ejemplo'})
my_app.add_error_handler(500, custom_error_handler)
CORS(my_app.app)
if __name__ == '__main__':
    my_app.run(port=8081)
