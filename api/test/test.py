import os
import requests
import json
import base64

url = 'http://localhost:8081/'

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

# Enum para los valores de lora
Loras = {
    "bridal_hairstyle-10.safetensors": "Largo",
    "curls_hairstyle-10.safetensors": "Largo",
    "dreads_hairstyle.safetensors": "Largo",
    "ponytail_weave_hairstyle.safetensors": "Largo",
    "egyptian_bob_hairstyle.safetensors": "Corto",
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
# Directorio que contiene las imágenes
directory = 'images'
output_directory = 'decoded_images'
# Iterar sobre cada lora
for lora_key, lora_value in Loras.items():
    # Iterar sobre cada archivo en el directorio
    for filename in os.listdir(directory):
        # Construir la ruta completa del archivo
        file_path = os.path.join(directory, filename)

        # Leer el contenido del archivo de imagen
        with open(file_path, 'rb') as file:
            image_content = file.read()

        # Codificar la imagen en base64
        base64_image = base64.b64encode(image_content).decode('utf-8')

        # Datos del cuerpo de la solicitud
        data = {
            'image': base64_image,
            'lora': lora_key,
            'seed': 22222222
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
