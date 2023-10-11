import cv2
import numpy as np
import base64


def process_image():
    #NO FUNCIONA
    base64_data_face = b"data:image/png;iVBORw0KGgoAAAANSUhEUgAAANsAAAFLCAAAAAB6+aZrAAAC/0lEQVR4nO3d2U7cQBAFUBzl/3/ZeQgCZl/6VncZnfMQZZSR3de3vAwi8PEBAAAAAAAAAAAAAAAAB7Pvy3a9FW9/n7GTG/7Ubn7/8ed0tYf0O9SK6kr3eaWvmRkr93V9FOelK9zTzbNsVrria8lVsy4tddnuJJgU7m/Rdtfdsb8V9fYg2pzkJef1E0ufcT2p2MdTrXzteC8LWjCTrw9c0fN0PNuz69zvvApJZ3t3kRXhwtleWOJ+92VCNttLC/x883b6MiiabWx58XDJbK8u7vz96ctlMNu7K/txd8uGy2WLrCsabsVnnC9XkiTDxbKlFhUMt7S34s8DqWwjq6z6TBDKFnzUynW5diZrZbK9f6wrz7jlve0fvT+bDh38uuaW91aoQbaLR+bUhhOjnp6q1OnXoLcLqWMVyNbhS8hXdewtdbTGs1XUltlmy95CZLuj7ZWkbW+RI9Y0W0TXbIniumZLkG2BwFCOZmt8C+jbW8Bvzjb2vTOdJ7Jzb+PHrW+2cUPZeo+k3g5KtiWGz+bG2YbJdsOi/4vyLL3d0ru4wd5ahzOTa4ze4DpnGyXbMcm2xugNpnO2UbIdk2zHJNsSw58xGmcbJtsxyXZb5y8q6O2YZFth/Ezum22cbMc0nK3xDU5vxzSere9Q6u2utsXpbYHANCSydR3Ktr0FyPZA06HU23yJUchk6zmUXXtLkO2h9FBGtqe3xzpeTXr2ljlSsWwNi+vZW0YuW7C40KaCvbWbyuRMdgvnfHtSs+KyvYXCtfm5MydaNZc+3zqFi19LGoXLXyf7/DSznveAZs/K39pMZUVvXcKVzORouH7Pyj/0aK7oWjIUrvvPaOzQXNk94P1wscNSd39b31zhvfvNcLljUvlc8tYqg3WXPnMtHsva58knw21X/zqs6nekfdoePNGXNjthbO7E287fE13OlFPiRrrt4i3Z1RTP5H/bq783LbbbiU4Sbmf/sv5mn7Dw9ycDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAb/EPDM1YCPOa3CkAAAAASUVORK5CYII"
    #FUNCIONA
    base64_data_hair = b"iVBORw0KGgoAAAANSUhEUgAAANsAAAFLCAAAAAB6+aZrAAAGS0lEQVR4nO2d2XbbMAxEpZ7+/y+7D00TL1wAYkAOeuY+JMeJRfISIKnFlq5LCCGEEEIIIYQQQgghhBBCCCHE/8q9sa7H5oo3uT1af8yuO92tabWn9l+ppU/UJv+N8juz8NymT0nKCrtVYlrm5KQjYImxzR5vcx5pdhkp4W9sTmImlLoUhww7fJmrKYZvCbrEwOCBy4ELjM0L4MZg58nglAeeMaFu4bZh5ZBugJZB5c6v3a8gV3KgG6hVODmcG6xNsIJgbsBcQhXFNt6u64LJodywszemNJAb+jgFUh5lTl4YOYzboCX3zjOgb1UjChmpTf4/2TJEck7eTz+dxLMSEbd+K27De/pE2wZws6gZ3mzb3kM8J11qW6eWxPHWsXDYBYdc2G2h/l2hy4vbwMAsFwtc1rWOcyv2D9G4dXp2orYncMH+bddtKNTc6EADM8abpTk/77nHE2cgcjE30GXspEUv5NZSs7bzHrwCgc7J1Ub2t1tPyohbo1aH2j18iQAbN1cDv978eH0JJOD2GbZY8+ByyLh5G/f+/s40tDzg1t0+qlzt96eCsKHDxQ3SLqjc0XN4DROkHMwN1ajoeYgnzp57Zf0cHm6naRqXxcCB4gbc1cLFkvV6wCtrgcOsb+t9nTnijsftvtI+ZwhxC3V+XuSOx83GUmSX3XB59LHLjKoOcX6S4HpLk9W4ZX6EHOUKGG97TiEvdCblXAIacYtuaceTgzLdcpRx6+CVq+TmlQu7MVyM6kAaN0iPkbp18CUlqxsicKxuCOT2zoYvJAKSMho34iVAOVmUpWPT7+HGnJHMcYsfC/C6xQm5HUhJT+BW3A5/3d6McpIIR9IsuO1KyfBoLhc3B/Xc7GkD/xweEfXiZu/RhM+G0uB3e3RfsFEwJ80wu/UWOGuyMLtFkVtN3G4EM6N1R5M5btFuZHaLIreayI0L7ZdQuw3CY4scsdsIk5zXjWC3xEzJuBk7uKTbZdNL+n43BVXjZunUsm4Gubpuczlet/hI5nWbM7Ov7DZDbgNOLnCTuhU3VsaBy7w/12m8bmSf4Bp2LCAnaQNXe7yNoXUzZsPobQi3o0k5qJw2bmb6cvXd+kDczs6U3dpTn2u0iUdn2WXNyRP3V27XeXz5bjbA6XZcwgNrTnppdbrPrRs23as9k0ZvuNzCV8T2whm3lZ763IbxeR0oPG6EzR9C+AyZ1aI+tnK4zWqkCyvZ86ig2N3YWj6H6xlpWKiebfcP0ElQ8NoNvLPVxuesAM6peaC8V8S5MffRHQl38oddMvB102e1nPvKCzR6NMHtSFY2n1dj3djYYug1rOA4AMft0OW5drXmc6833W7HFG9Hj/0OPZ24U23xa8JD3OONUK7XpP9mfWsgt5owu0V3F/xuo8lkcQVMWjgPXzcFWPX7+mxOPt5+Yznp9vSM9XW5wRAhnkvCwTzolr7rfWguMXjF1cFuD/PT3zCPjRhybA34tsvb+V5wu7nOHvdBzyV5j0zwQ7wGhJGbA6ITRorbK4SnTJrg48aTlMrJN/Ykpa2WvZ+hp0nK8jlZ89g0zJpbjVVAcfPBMplkuLFkrHKyJotuLGk3hHguqXzuNR25nYD03iwPjtU7KW4Udmk5GZUDdA7veLMxWmiru43Iczs/4hQ3Wob7tYluoaQ88d31f1Q4EMjMydOzSe3xNs6eVTdTTA4HrnTcJoM+1205cBXuOZCalbO5WmvAOienk9JzyYR0t6XAYaK97FZgwOXn5EIMQIN03c38JeW06WRWMOG9x2B9oXmyTU5S4r5srbhRMk2bwm5Ttrh5Bpz5vfPRHnIj3zcpm5OGft3jZk/K4Fe6XygbNwMxN+4BRxY3ZEqyuUEJupm/EmYDezhUM262vtridmjKST42zbEylprpdnqFSMzJNLXzz0peUTNtYy443LndaTtQMuimRGlxi3Qa5z0aL0zDMHIJcbu/fwTLcP/rDbjbfV33He/4bgGOksNuzbpOr2x/od2f7HSPp9do3dq4EqKWmy/Xed10P4URcbeNc6KzKsWtJgC3rKRcvxf0F4Xi5u7DQm5uEG5Mt1d4BhK3JLm7+2Jh+3V+zgJwHAJc16XxZoAoWj+g4sYoB8vJ++UXBci29J5XKYQQQgghhBBCCCGEEEIIIYQQQqD5A9TT3gnBb43qAAAAAElFTkSuQmCC"

    imagen_face = cv2.bitwise_not(cv2.imread("face.png"))
    image_hair = get_image_from_base64(base64_data_hair)



    # get_lower_point(image)
    # dibujar_punto_centro(image)
    image = ensanchar_borde(image_hair, 25)
    limpiar_cara(imagen_face, image)

    cv2.imshow("Imagen Original", image)
    cv2.imshow("Imagen Hair", image_hair)

    cv2.waitKey(0)


def limpiar_cara(imagen_face, image):
    # Iterar sobre los píxeles de la imagen 1
    for y in range(imagen_face.shape[0]):
        for x in range(imagen_face.shape[1]):
            if all(imagen_face[y, x] == 0):  # Verificar si el píxel es negro
                image[y, x] = 255  # Copiar el píxel de la imagen 1 a la imagen 2

    # Guardar la imagen resultante
    cv2.imwrite("imagen_resultado.png", image)


def get_image_from_base64(base64string):
    imagen_bytes = base64.b64decode(base64string)
    # Convertir los bytes a una matriz NumPy (esto representa la imagen)
    imagen_np = np.frombuffer(imagen_bytes, np.uint8)
    # Decodificar la imagen usando OpenCV
    imagen = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)
    imagen = cv2.bitwise_not(cv2.imdecode(imagen_np, cv2.IMREAD_GRAYSCALE))
    return imagen

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

def ensanchar_borde(imagen, dilatacion):
    # Invertir los colores (negativo)
    imagen_invertida = cv2.bitwise_not(imagen)

    # Definir el kernel para la operación de dilatación
    kernel = np.ones((dilatacion, dilatacion), np.uint8)

    # Aplicar la operación de dilatación
    borde_ensanchado = cv2.dilate(imagen_invertida, kernel, iterations=1)

    # Invertir nuevamente los colores para obtener el resultado final
    borde_ensanchado = cv2.bitwise_not(borde_ensanchado)

    # Mostrar la imagen original y la imagen con el borde ensanchado
    # cv2.imshow("Imagen Original", imagen)
    # cv2.imshow("Borde Ensanchado", borde_ensanchado)

    return borde_ensanchado



def dibujar_punto_centro(imagen):
    # Obtener las dimensiones de la imagen
    alto, ancho = imagen.shape

    # Calcular las coordenadas del centro
    centro_x = ancho // 2
    centro_y = alto // 2

    # Establecer el color negro (en formato BGR)
    color_negro = (0)

    # Asignar el color negro al píxel en el centro
    imagen[centro_y, centro_x] = color_negro

    # Mostrar la imagen actualizada
    cv2.imshow("Imagen con Píxel Negro en el Centro", imagen)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process_image()

