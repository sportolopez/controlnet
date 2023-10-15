import connexion

def generar(body):
    imgBase64 = body['imgBase64']
    prompt = body['prompt']
    neg_prompt = body['neg_prompt']

    # Aquí debes implementar la lógica para procesar los parámetros y generar la imagen en base64.
    # Agrega tu lógica de generación de imágenes aquí y devuelve la imagen generada en base64.

    # En este ejemplo, simplemente devolveré un mensaje de éxito junto con los parámetros.
    return {'message': 'Imagen generada con éxito', 'imgBase64': imgBase64, 'prompt': prompt, 'neg_prompt': neg_prompt}

if __name__ == '__main__':
    app = connexion.App(__name__, specification_dir='./')
    app.add_api('swagger.yaml', arguments={'title': 'API de Ejemplo'})
    app.run(port=8081)