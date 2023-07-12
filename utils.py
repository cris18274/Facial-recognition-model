import cv2

def load_image(image_path):
    # Carga una imagen desde el archivo
    image = cv2.imread(image_path)
    return image

def detect_faces(image):
    # Carga el archivo de cascada frontal de Haar pre-entrenado para la detección de rostros
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecta las caras en la imagen
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Retorna las coordenadas de las caras detectadas
    return faces

def draw_faces(image, faces):
    # Dibuja un rectángulo alrededor de cada cara detectada en la imagen
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Muestra la imagen resultante
    cv2.imshow('Reconocimiento Facial', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
