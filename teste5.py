import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import time
import mediapipe as mp
import os

# Preprocessamento de Imagens
def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    return resized_image / 255.0

# Criar modelo de classificacao
def create_classification_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Classificar frame
def classify_frame(frame, model):
    preprocessed_frame = preprocess_image(frame)
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    return prediction

# Criar pasta para armazenar as imagens se nao existir
def create_images_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Inicializar deteccao de maos
def initialize_hand_detection():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Inicializar deteccao de rostos
def initialize_face_detection():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar tracker de rostos
def initialize_face_tracker():
    return cv2.TrackerKCF_create()

# Captura de video da webcam
def capture_webcam_video(index=0):
    return cv2.VideoCapture(index)

# Main function
def main():
    num_classes = 3
    model = create_classification_model(num_classes)
    face_cascade = initialize_face_detection()
    hands = initialize_hand_detection()
    tracker = initialize_face_tracker()
    cap = capture_webcam_video()

    start_time = time.time()
    interval = 3
    images_folder_path = r"C:\Users\eliak\visaocomputacional\imagens"
    create_images_folder(images_folder_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar o frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            if 'face_box' not in locals():
                face_box = (x, y, w, h)
                ok = tracker.init(frame, face_box)

            ok, face_box = tracker.update(frame)
            if ok:
                x, y, w, h = [int(i) for i in face_box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Rosto", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if time.time() - start_time >= interval:
                start_time = time.time()
                frame_to_recognize = frame[y:y+h, x:x+w]
                prediction = classify_frame(frame_to_recognize, model)
                print("Predição:", prediction)

                autorizado = True

                if autorizado:
                    nome_imagem = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
                    caminho_imagem = os.path.join(images_folder_path, nome_imagem)
                    cv2.imwrite(caminho_imagem, frame_to_recognize)
                    print("Imagem salva:", caminho_imagem)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x:
                    mao_texto = "Mão Direita"
                else:
                    mao_texto = "Mão Esquerda"

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                cv2.putText(frame, mao_texto, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
