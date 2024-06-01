import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Carregar o modelo treinado
model_path = 'user_recognition_model_final.keras'
model = load_model(model_path)

# Caminho do dataset para obter os nomes das classes
dataset_path = 'dataset/'
class_names = os.listdir(dataset_path)

def recognize_face(frame, face_cascade, model, class_names):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Fazer a predição
        predictions = model.predict(face)
        max_index = np.argmax(predictions[0])
        confidence = predictions[0][max_index]
        
        if confidence > 0.5:  # Ajuste o limiar de confiança conforme necessário
            label = class_names[max_index]
            text = f"{label}: {confidence:.2f}"
        else:
            label = "Desconhecido"
            text = f"{label}: {confidence:.2f}"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return frame

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha na captura do frame.")
            break

        frame = recognize_face(frame, face_cascade, model, class_names)
        cv2.imshow('Reconhecimento Facial', frame)

        if cv2.waitKey(1) % 256 == 27:  # Pressione ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
