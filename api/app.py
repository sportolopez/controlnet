import base64
import json
import logging
import re
import time
import traceback
from io import BytesIO
import jwt
import connexion
import requests
from PIL import Image
from flask import send_file, request, jsonify
from flask_cors import CORS
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from controlnet.HairSegmentation import segment_hair
from controlnet.ControlNetSegment import ControlNetSegment
from controlnet.Loras import Loras

controlnet = ControlNetSegment()

def get_image_from_base64(base64string):
    imagen_bytes = base64.b64decode(base64string)
    image_buffer = BytesIO(imagen_bytes)
    image = Image.open(image_buffer)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

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
    inicio_total = time.time()
    body = connexion.request.get_json()
    imagen_base64 = body['image']
    if(imagen_base64.startswith('data:image/')):
        image = get_image_from_base64(body['image'].split(",")[1])
    else:
        image = get_image_from_base64(body['image'])

    prompt_default = '4k, high-res, masterpiece, best quality,((Hasselblad photography)), finely detailed skin, sharp focus, (cinematic lighting), soft lighting, dynamic angle'
    neg_prompt_default = '(greyscale:1.2),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans'
    key_word = ''

    if 'color' in body and body['color']:
        color = (body['color'])
    else:
        color = ""


    if 'lora' in body and body['lora']:
        lora = (body['lora'])
    else:
        lora = None

    if 'max_size' in body and body['max_size']:
        max_size = (body['max_size'])
    else:
        max_size = 2048


    if 'seed' in body  and body['seed']:
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

    bool_pelo_largo = False
    if lora:
        bool_pelo_largo = Loras.get(lora, "")
        key_word = re.sub(r'[^a-zA-Z]', '', lora.split('.')[0])

    prompt = color + ", " + prompt + " , " + key_word

    print("*****Realizando prueba******")
    print("prompt:"+prompt)
    if lora:
        print("Lora: "+lora)
    print("Neg_prompt:" + neg_prompt)
    print("max_size:" + str(max_size))
    print("Seed:" + str(seed))

    if 'imagen_mask' in body:
        imagen_mask = get_image_from_base64(body['imagen_mask'])
    else:
        imagen_mask = segment_hair(image,bool_pelo_largo)





    inicio = time.time()



    #image = resize_image_if_big_by_size(image,max_size)
    #imagen_mask = resize_image_if_big_by_size(imagen_mask, max_size)
    imagen_gen = controlnet.segment_generation(image=image, image_segm=imagen_mask, prompt=prompt,
                                               neg_prompt=neg_prompt, seed=seed, lora=lora)

    image_bytes = BytesIO()
    imagen_gen.save(image_bytes, format="PNG")  # You can choose a different format if needed
    image_bytes.seek(0)
    inicio = time.time()
    imagen_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
    print(f"****La ejecución de  base64.b64encode  tardó { time.time() - inicio} segundos")
    print(f"****La ejecución de  TOTAL  tardó {time.time() - inicio_total} segundos")
    # En este ejemplo, simplemente devolveré un mensaje de éxito junto con los parámetros.
    return "data:image/png;base64,"+imagen_base64


def status():
    # Aquí debes implementar la lógica para procesar los parámetros y generar la imagen en base64.
    # Agrega tu lógica de generación de imágenes aquí y devuelve la imagen generada en base64.
    # foto
    # lora
    # color
    # 
    # En este ejemplo, simplemente devolveré un mensaje de éxito junto con los parámetros.
    return {'message': 'Prueba'}

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
        if(token_info is None):
            return jsonify({'error': "Access token invalido"}), 401
        print(token_info)
        # Generar JWT
        jwt_token = jwt.encode({'token_info': token_info}, 'secret_key', algorithm='HS256')

        return jsonify({'jwt_token': jwt_token, 'user_info':token_info}), 200
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
