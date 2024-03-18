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
    "bridal_hairstyle-10.safetensors": "bridal_hairstyle",
    "curls_hairstyle-10.safetensors": "curls_hairstyle",
    "dreads_hairstyle.safetensors": "dreads_hairstyle",
    "ponytail_weave_hairstyle.safetensors": "ponytail_weave_hairstyle",
    "egyptian_bob_hairstyle.safetensors": "egyptian_bob_hairstyle",
    "emo_hairstyle.safetensors": "emo_hairstyle",
    "long_braid_hairstyle-10.safetensors": "long_braid_hairstyle",
    "middle_parting_hairstyle.safetensors": "middle_parting_hairstyle",
    "short_dreads_hairstyle.safetensors": "short_dreads_hairstyle",
    "a_line_hairstyle.safetensors": "a_line_haircut",
    "baldie_hairstyle.safetensors": "baldie_hairstyle",
    "bun_hairstyle-10.safetensors": "bun_hairstyle",
    "buzzcut_hairstyle.safetensors": "buzzcut_heairstyle",
    "colored_buzzcut_hairstyle-10.safetensors": "colored_buzzcut_hairstyle",
    "half_buzzcut_hairstyle.safetensors": "half_buzzcut_heairstyle",
    "half_ponytail_hairstyle-10.safetensors": "half_ponytail_hairstyle",
    "hi_top_fade_hairstyle.safetensors": "hi_top_fade_hairstyle",
    "knotless_braid_hairstyle.safetensors": "knotless_braid_hairstyle",
    "long_hime_cut_hairstyle.safetensors": "long_hime_cut_hairstyle",
    "long_ponytail_hairstyle.safetensors": "long_ponytail_hairstyle",
    "pigtail_hairstyle.safetensors": "pigtail_hairstyle",
    "pixie_hairstyle-05.safetensors": "pixie_hairstyle",
    "short_pigtail_hairstyle05.safetensors": "short_pigtail_hairstyle",
    "side_buns_hairstyle.safetensors": "side_buns_hairstyle",
    "side_swept_hair-05.safetensors": "side_swept_hair",
    "space_buns_hairstyle.safetensors": "space_buns_hairstyle",
    "updo_hairstyle.safetensors": "updo_hairstyle",
    "very_long_hair-10.safetensors": "very_long_hair"
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
            'lora': lora_key
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
