import cv2
import numpy as np
import base64
import datetime


def hora():
    hora_inicio = datetime.datetime.now()
    hora_inicio_formateada = hora_inicio.strftime('%H:%M:%S.%f')[:-3]
    print(f'Hora: {hora_inicio_formateada}')


def process_image():
    hora()
    base64_skin = "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"
    base64_ceja_r = "iVBORw0KGgoAAAANSUhEUgAABAAAAAMACAAAAACf0UrRAAAESklEQVR4nO3dy1ICUQwE0Nb//+dxIaUWgvMguXPBc1aUKyrdCSxAEgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAeAFvZz+Bf2T5fmjszEETx1hu/dHwOZsO9ru5/Bfmz6kUsN1f+x8JcCr167Wy/Z+EwFl0r9Om9U8iB06ieH22r38SUXAGrWuzc/8TaTCcynU5sP+JQBhL35oc3P8kQmEcXevxyP4ncmEQRWvx6P4nomEELWtQsf5JpEM7FatXtv9JJEQr9SpXu/+REY3ez34CrFrKTwpceHWp1rOtcqKFdwDPwbsAWnhlqXZjU6+H7EOCzEKpil0t9/35HrkC0qKYShX7uderw919BMRFLY2q9rXUG0e77wjIi1IKVe2y0LsGu+cISIxC6lRuyaGxbj4CIqOONtVbDg7VCWA4XZrJ1hMgNYqo0mS23QCxUUOTJrR+BMRGDU2ak18TYQhNmtafN0BulFCkqd09AnKjhCJNzw8L00eRnsKvIyA3SijS09j1NSPgNfnfIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8Aw+AIguLR89NQRhAAAAAElFTkSuQmCC"
    base64_ceja_l = "iVBORw0KGgoAAAANSUhEUgAABAAAAAMACAAAAACf0UrRAAAENUlEQVR4nO3dzU6DUBgE0Kvx/V8ZFxiLCJV//DrnrIyLK8nMIDVVWwMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA/+7t7gugim7wsdq8CkmyQDfxOdV5BVLkuantP+hPcQJkzvPpDyhRXbJjyuLx99SoKskxtnL8XzSpJLHxsG3635SpHpnR2zn+njpVIzEOGn9PoWqRV7gDt9/TqFLEFezw8bfWVKoWacU6Z/5Np0oRVqjT5t+UqpKPuy+AO5w5fypxsw50+vy1qoz3uy+Ay53/7d8DRhleAqQxTgbcALJsnv/4qd595DV4tRZl02znOzJ/nF4VIaggB8//2al6VYSgcmzY/+J6jM/WqyIEFWP9/teV4+f5ilWDnFJc8Td+Hl9Dr4oQVIo1N4A9rej2HsCVJBVi4f6P6EOnVXWIKsSCG4AuBBJ6hj/2rwapvBMwnvEnk36I6UcA8afTgBTeqsMENYjhv/vym58BxDF+SNT5FV4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALjZJ0onKSxh1+Q5AAAAAElFTkSuQmCC"
    base64_nose = "iVBORw0KGgoAAAANSUhEUgAABAAAAAMACAAAAACf0UrRAAAFFElEQVR4nO3cwVLCMBiF0eD7v7MuBAVstdpUvN5zlg6LziT58hcdxwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOB3nR79AEz1PMawqmxmq/wHzys/t7p8wRaJt3b6z6wwn7A90n1x/sewyKx7evQDsM+G87/pM3RyOWTberatM4tMANE23+2GABYJABQTgGTfuNeNACwRACgmAFBMAIIZ69lLAFqoBQsEAIoJABQTACgmAFBMAKCYAOTyvT67CQAUEwAoJgBQTACgmABAMQGAYgIQy28B2U8AoJgAQDEBgGICAMUEAIoJABQTACgmAFBMAKCYAEAxAajhT4f5SABSOc9MIABQTACgmABAMQEI5SsAZhAAKCYAUEwAMnkDYAoB6CEafCAAkZxl5hAAKCYAUEwAEnkDYBIBgGICEOinA4DBgXsCAMUEAIoJABQTgDxe5ZlGAKCYAMTZMQCYHbgjAFBMAKCYAKQxxjORAEAxAahifOCWAITZeYQVgBsCAMUEIMvuG9wIwDUBgGICEGXC/W0E4IoAQDEBSOL2ZjIBgGICEGTOAGCM4J0AQDEByOHqZjoBgGICEGPaAGCS4I0ApHBsOYAAQDEBCGEA4AgCUEhMuBCADM4shxCARnLCmQBAMQGI4MrmGAJQSVB4JQAJnFcOIgBQTACgmAB08lLBGEMAoJoAQDEBCGBe5ygCAMUEoJSpgjEEIIGzymEEAIoJABQTACgmAFBMAKCYAEAxAYBiAgDFBACKCUCp06MfgD9BAP4+Z5XDCEAnUWGMIQBQTQACzL+uDQC8EoAEzisHsbVCTP2nAFadM1shxrwEWHQu7IUgkxJgzXljM4TZHwFLzju7Ic+uBlhwrtkPsX7UAevNDRsi23oFTgufsNrcsSX+gbsKWFMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABY8QJthz62rfbqXQAAAABJRU5ErkJggg=="
    base64_labio_s = "iVBORw0KGgoAAAANSUhEUgAABAAAAAMACAAAAACf0UrRAAADzklEQVR4nO3ZQQqDQBAEwDH//7M5aFDQCJp1nc1WHXMIQvc0hkQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwsOHpB+CscfOJELlKd9qyvf6FLDlNaRpydP1fSZgD6tGIS8e/kDO7FKMFP17/TNZsKEV+Zc4/QtpsqERC4yqVctc/kzgr6pDPfPND+eOPzzfDRBfyuevu1+RORES8nn4AHjHWWBnyMwC9sgCEV8Gk/AqgDi1IqtoEjErQM9knVu8tXQ16Jfn0qsyAHnRK8I24ewYUoU9yb8uNO6AKPZJ6i+6ZAV3okNCbVmQJpj8DNKFLYv8fZ9dA9ijBP9tdBIkDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAhbwBuUgaFSu3iGsAAAAASUVORK5CYII="
    base64_labio_i = "iVBORw0KGgoAAAANSUhEUgAABAAAAAMACAAAAACf0UrRAAAESElEQVR4nO3c0XKCMBAF0LTT//9lfGinKkYFTcLGPecDlJndewFhLAUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgP2+jj4AVpZSjIVRvo8+AGqW5egjIAenmnDO2TccerNj8azO/kZEP7YrosoNgEHRg70K6e5PAOZFUxYqpsc/ApoajVilmDY+BWgyPg8eEzP3oF54Drh3lpdfYQ+SMvioGrwJUB1u7XNtQVpGH9W4V4HsQGLeBIxqWCzlPzMFENagYMp/agogOfnPTQFAYgogrhEnZxcAySmA1OQ/OwWQmfynpwAC651P+UcBQGIKILK+p2gXACiA2HpmVP5RANH1S6n8UxRAWvJPKQogvE5BlX9KKQogvi5RlX9+KYDwOoRV/vnzc/QBMJz4888yzKDpvwMZOWduAWbQMrPyzwXrMIlGFwHmzRULMY0WFWDcXLMRE3m3AgybNTsxlzc6wKi5ZSum81IHmDNVFmNGezvAlLnDasxqawmYMA9Yj6k9aQHT5Qkr8hFuisBcAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC2OwEAMxlWbj9YXwAAAABJRU5ErkJggg=="
    base64_diente = "iVBORw0KGgoAAAANSUhEUgAABAAAAAMACAAAAACf0UrRAAAESElEQVR4nO3c0XKCMBAF0LTT//9lfGinKkYFTcLGPecDlJndewFhLAUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgP2+jj4AVpZSjIVRvo8+AGqW5egjIAenmnDO2TccerNj8azO/kZEP7YrosoNgEHRg70K6e5PAOZFUxYqpsc/ApoajVilmDY+BWgyPg8eEzP3oF54Drh3lpdfYQ+SMvioGrwJUB1u7XNtQVpGH9W4V4HsQGLeBIxqWCzlPzMFENagYMp/agogOfnPTQFAYgogrhEnZxcAySmA1OQ/OwWQmfynpwAC651P+UcBQGIKILK+p2gXACiA2HpmVP5RANH1S6n8UxRAWvJPKQogvE5BlX9KKQogvi5RlX9+KYDwOoRV/vnzc/QBMJz4888yzKDpvwMZOWduAWbQMrPyzwXrMIlGFwHmzRULMY0WFWDcXLMRE3m3AgybNTsxlzc6wKi5ZSum81IHmDNVFmNGezvAlLnDasxqawmYMA9Yj6k9aQHT5Qkr8hFuisBcAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC2OwEAMxlWbj9YXwAAAABJRU5ErkJggg=="
    base64_ojo_r = "iVBORw0KGgoAAAANSUhEUgAABAAAAAMACAAAAACf0UrRAAAD2klEQVR4nO3aSwrCUBAEwNH73zluYpCgmPcQjT1VO8GF0D3tB6sAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+LzLr18AZ7VoRwMiZm/ZPdaRYMLl0f747/QklGDZvLr+la4EEiqrN+dfVeqS5/rrF8BJHLn/Y0/ij5h0qkYuW2OiiJPRN3adCeIrAKMf7H0PCGIAGD5oC5DDALQ3cc4WIIYB6G7qmC1ACgPQ3OQpW4AQBoApFiCDAeht/o4tQAQDAI0ZAOb4O1AEA9CbM27OADQ3uwCWI4MB6M4ltyZ+Zn7Q15sQgqTGJ0BtUkiSqhqbAKXJIUtWBydAY6KIkwdvR0BfwgiUnZcjoCuBhMpT2wxoCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8F03zKkdH3Ov0pwAAAAASUVORK5CYII="
    base64_ojo_l = "iVBORw0KGgoAAAANSUhEUgAABAAAAAMACAAAAACf0UrRAAAD0ElEQVR4nO3cQQ7CMAwEwML//wwHxAlE0yJvDJk5oh4i2d4YhLptAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMx0mX2Add1eP1INwrTcFG+G/0lFCNJueR+m/0FRSNFrcbvzvykLKdfZB1jOyPyPPQRfc9VkjU+2yhBgA4g6cLNbAghwzyQdHGrFoZoNoDFLANUEQNDhgZYAFBMArUkAagmA3iQApQRAcxKASgKgOwlAIQHQngSgjgDoTwJQRgDAwgRAkH/20Y0A+AG+A1BFACSdXQEkAEUEACxMAET5FYBedGTaiXVekaiit/K8FYA2dNcUoxmgPNTSYfPspIDSUE+XdfGMAxUBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+0R3D+xogfDInJwAAAABJRU5ErkJggg=="

    image_skin = get_image_from_base64(base64_skin)
    image_ceja_r = get_image_from_base64(base64_ceja_r)
    image_ceja_l = get_image_from_base64(base64_ceja_l)
    image_nose = get_image_from_base64(base64_nose)
    image_labio_s = get_image_from_base64(base64_labio_s)
    image_diente = get_image_from_base64(base64_diente)
    image_labio_i = get_image_from_base64(base64_labio_i)
    image_ojo_r = get_image_from_base64(base64_ojo_r)
    image_ojo_l = get_image_from_base64(base64_ojo_l)

    imagen_unida = formar_cara_completa(image_skin, image_ceja_r, image_ceja_l, image_nose, image_labio_s, image_labio_i, image_diente, image_ojo_r, image_ojo_l)
    image = ensanchar_borde(imagen_unida, 150)
    lowerpoint = get_lower_point(image_labio_i)
    image_clean = get_a_line_haircut(imagen_unida, image, image_ceja_r, image_ceja_l, lowerpoint)
    cv2.imshow("Imagen Original", ensanchar_borde(imagen_unida, 150))

    hora()
    cv2.waitKey(0)


def formar_cara_completa(image_skin, image_ceja_r, image_ceja_l, image_nose, image_labio_s, image_labio_i, image_diente, image_ojo_r, image_ojo_l):
    # Aplicar la operación bitwise OR
    imagen_unida = cv2.bitwise_and(image_skin, image_ceja_r)
    imagen_unida = cv2.bitwise_and(imagen_unida, image_ceja_l)
    imagen_unida = cv2.bitwise_and(imagen_unida, image_nose)
    imagen_unida = cv2.bitwise_and(imagen_unida, image_labio_s)
    imagen_unida = cv2.bitwise_and(imagen_unida, image_diente)
    imagen_unida = cv2.bitwise_and(imagen_unida, image_labio_i)
    imagen_unida = cv2.bitwise_and(imagen_unida, image_ojo_r)
    imagen_unida = cv2.bitwise_and(imagen_unida, image_ojo_l)
    return imagen_unida


def get_a_line_haircut(imagen_face, image, image_ceja_r, image_ceja_l, lower_point):
    coordenadas = cv2.minMaxLoc(image_ceja_r)
    x, y_ceja_r = coordenadas[2]
    coordenadas = cv2.minMaxLoc(image_ceja_l)
    x, y_ceja_l = coordenadas[2]
    y_ceja = y_ceja_r if y_ceja_r > y_ceja_l else y_ceja_l

    # Iterar sobre los píxeles de la imagen 1
    for y in range(imagen_face.shape[0]):
        for x in range(imagen_face.shape[1]):
            if (imagen_face[y, x] == 0 and #Elimina la cara
                    # Agrega la frente
                    y > y_ceja - 10) or \
                    y > lower_point[0]: #Corte hasta un Y determinado (obtenido por el borde de la boca)
                image[y, x] = 255  #
    return image


def get_image_from_base64(base64string):
    imagen_bytes = base64.b64decode(base64string)
    # Convertir los bytes a una matriz NumPy (esto representa la imagen)
    imagen_np = np.frombuffer(imagen_bytes, np.uint8)
    # Decodificar la imagen usando OpenCV
    imagen = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)
    imagen = cv2.bitwise_not(cv2.imdecode(imagen_np, cv2.IMREAD_GRAYSCALE))
    return imagen


def get_higher_point(imagen):
    # Obtener las dimensiones de la imagen
    alto, ancho = imagen.shape

    # Inicializar las coordenadas del punto más bajo
    x_masalto = 1000000
    y_masalto = 1000000
    # cv2.imshow("Imagen Original", cv2.bitwise_not(imagen))
    # Iterar sobre las filas desde abajo hacia arriba
    for y in range(0, alto):
        for x in range(ancho):
            if imagen[y, x] == 0:  # Si el píxel es negro (0 en escala de grises)
                x_masalto = x
                y_masalto = y
                break
        if x_masalto != 1000000:
            break

    # Verificar si se encontró un píxel negro (forma irregular presente)
    if x_masalto != 1000000:
        print(f"El punto más bajo está en las coordenadas: ({x_masalto}, {y_masalto})")
    else:
        print("No se encontró una forma irregular en la imagen.")
    return x_masalto, y_masalto


def get_lower_point(imagen):
    # Obtener las dimensiones de la imagen
    alto, ancho = imagen.shape

    # Inicializar las coordenadas del punto más bajo
    x_masbajo = -1
    y_masbajo = -1

    # Iterar sobre las filas desde abajo hacia arriba
    for y in range(alto - 1, -1, -1):
        for x in range(ancho):
            if imagen[y, x] == 0:  # Si el píxel es negro (0 en escala de grises)
                x_masbajo = x
                y_masbajo = y
                break
        if x_masbajo != -1:
            break

    # Verificar si se encontró un píxel negro (forma irregular presente)
    if x_masbajo != -1:
        print(f"El punto más bajo está en las coordenadas: ({x_masbajo}, {y_masbajo})")
    else:
        print("No se encontró una forma irregular en la imagen.")
    return x_masbajo, y_masbajo


def ensanchar_borde(imagen, dilatacion):
    # Invertir los colores (negativo)
    imagen_invertida = cv2.bitwise_not(imagen)

    # Definir el kernel para la operación de dilatación
    kernel = np.ones((dilatacion, dilatacion), np.uint8)

    # Aplicar la operación de dilatación
    borde_ensanchado = cv2.dilate(imagen_invertida, kernel, iterations=1)

    # Invertir nuevamente los colores para obtener el resultado final
    borde_ensanchado = cv2.bitwise_not(borde_ensanchado)

    return borde_ensanchado


if __name__ == '__main__':
    process_image()

