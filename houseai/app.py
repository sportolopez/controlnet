import json
import logging
import traceback
import uuid

from flask_cors import CORS
import connexion


def login():
    # Implement your login logic here
    return "Token válido", 200

def generar(requestBody):
    token = requestBody.get("token")
    generation_id = requestBody.get("generation_id")
    image = requestBody.get("image")
    theme = requestBody.get("theme")
    room_type = requestBody.get("room_type")

    # Implement your logic to generate something based on the image
    return {"generation_id": uuid.uuid4()}, 200

def consultar(requestBody):
    generation_id = requestBody.get("generation_id")

    # Implement your logic to consult based on the generation_id
    return {"image_url": "https://comomequeda.com.ar/generated_img/00.jpeg", "remaining_photos": 5}, 200


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
if __name__ == '__main__':
    my_app.run(port=8082)
