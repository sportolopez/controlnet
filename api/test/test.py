import io
import os
import requests
import json
import base64
from PIL import Image, ExifTags

url = 'http://localhost:8081/'

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

# Enum para los valores de lora
'''
Loras = {
    "egyptian_bob_hairstyle.safetensors": "Corto"
}
'''
Loras = {
    "egyptian_bob_hairstyle.safetensors": "Corto",
    "bridal_hairstyle-10.safetensors": "Largo" ,
    "curls_hairstyle-10.safetensors": "Largo",
    "dreads_hairstyle.safetensors": "Largo",
    "ponytail_weave_hairstyle.safetensors": "Largo",
    "emo_hairstyle.safetensors": "Largo",
    "long_braid_hairstyle-10.safetensors": "Largo",
    "middle_parting_hairstyle.safetensors": "Largo",
    "short_dreads_hairstyle.safetensors": "Corto",
    "a_line_hairstyle.safetensors": "Corto",
    "baldie_hairstyle.safetensors": "Corto",
    "bun_hairstyle-10.safetensors": "Corto",
    "buzzcut_hairstyle.safetensors": "Corto",
    "colored_buzzcut_hairstyle-10.safetensors": "Corto",
    "half_buzzcut_hairstyle.safetensors": "Corto",
    "half_ponytail_hairstyle-10.safetensors": "Largo",
    "hi_top_fade_hairstyle.safetensors": "Corto",
    "knotless_braid_hairstyle.safetensors": "Largo",
    "long_hime_cut_hairstyle.safetensors": "Largo",
    "long_ponytail_hairstyle.safetensors": "Largo",
    "pigtail_hairstyle.safetensors": "Largo",
    "pixie_hairstyle-05.safetensors": "Corto",
    "short_pigtail_hairstyle05.safetensors": "Corto",
    "side_buns_hairstyle.safetensors": "Largo",
    "side_swept_hair-05.safetensors": "Largo",
    "space_buns_hairstyle.safetensors": "Corto",
    "updo_hairstyle.safetensors": "Corto",
    "very_long_hair-10.safetensors": "Largo"
}

def add_sufix_filename(ruta_completa, sufijo):
    carpeta, nombre_archivo = os.path.split(ruta_completa)
    nombre_base, extension = os.path.splitext(nombre_archivo)
    nuevo_nombre = f"{nombre_base}{sufijo}{extension}"
    nueva_ruta_completa = os.path.join(carpeta, nuevo_nombre)
    return nueva_ruta_completa


# Directorio que contiene las imágenes
directory = 'images'
output_directory = 'decoded_images'

archivos = os.listdir(directory)
archivos = [archivo for archivo in archivos if "_segm" not in archivo]
base64_image = ""
buffer = io.BytesIO()
# Iterar sobre cada lora
for lora_key, lora_value in Loras.items():
    # Iterar sobre cada archivo en el directorio
    for filename in archivos:

        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as file:
            # Abrir la imagen con PIL
            image_content = file.read()
            base64_image = base64.b64encode(image_content).decode('utf-8')

            with Image.open(file) as img:
                # Obtener el ancho y alto de la imagen
                if hasattr(img, '_getexif') and img._getexif():
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation] == 'Orientation':
                            break
                    exif = dict(img._getexif().items())

                    # Rotar la imagen si es necesario
                    if exif.get(orientation) == 3:
                        img = img.transpose(Image.ROTATE_180)
                    elif exif.get(orientation) == 6:
                        img = img.transpose(Image.ROTATE_270)
                    elif exif.get(orientation) == 8:
                        img = img.transpose(Image.ROTATE_90)

                    img.save(buffer, format="PNG")  # Guardar la imagen en un buffer de bytes
                    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Convertir la imagen a base64


        '''
        file_path_segm = add_sufix_filename(file_path, "_segm")
        with open(file_path_segm, 'rb') as file:
            image_content_segm = file.read()
        base64_image_segm = base64.b64encode(image_content_segm).decode('utf-8')
        '''



        # Datos del cuerpo de la solicitud
        data = {
            'image': base64_image,
            #'imagen_mask': base64_image_segm,
            'lora': lora_key
            #,'seed': 22222222
            ,'prompt': '4k, high-res, masterpiece, best quality,((Hasselblad photography)), finely detailed skin, sharp focus, (cinematic lighting), soft lighting, dynamic angle'
            ,'neg_prompt': '(greyscale:1.2),paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, shadows, background'
        }

        # Convertir el diccionario en formato JSON
        data_json = json.dumps(data)
        print(f'Enviando pedido  para {filename} y lora {lora_key}.')
        # Realizar la solicitud POST
        response = requests.post(url + '/generar', headers=headers, data=data_json)

        # Verificar el código de estado de la respuesta
        if response.status_code == 200:
            # Decodificar la respuesta base64
            decoded_content = base64.b64decode(response.text.split(',')[1])

            # Guardar el contenido decodificado en un archivo
            output_file_path = os.path.join(output_directory, f'{filename}_{lora_key}.png')
            with open(output_file_path, 'wb') as output_file:
                output_file.write(decoded_content)

            print(f'Respuesta exitosa para {filename} y lora {lora_key}. Archivo guardado en {output_file_path}')
        else:
            print(f'Error en la solicitud para {filename} y lora {lora_key}:')
            print(response.text)
