import cv2
import utils

def main():
    # Carga el archivo de cascada frontal de Haar pre-entrenado para la detección de rostros
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Inicializa la cámara (puedes cambiar el número 0 por la ruta de un archivo de video si deseas procesar un video en lugar de la cámara)
    cap = cv2.VideoCapture(0)

    while True:
        # Captura el marco de la cámara
        ret, frame = cap.read()

        # Convierte la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta las caras en la imagen
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Dibuja un rectángulo alrededor de cada cara detectada
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Muestra el marco resultante
        cv2.imshow('Reconocimiento Facial', frame)

        # Espera a que se presione la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
