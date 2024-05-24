import cv2
import numpy as np
import tensorflow as tf
import time
import mediapipe as mp
import os

# Função de pré-processamento de imagens
def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    return normalized_image

# Função de classificação
def classify_frame(frame, model):
    preprocessed_frame = preprocess_image(frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    prediction = model.predict(preprocessed_frame)
    return prediction

# Carregar o modelo treinado
model = tf.keras.models.load_model('user_recognition_model.h5')

# Carregar o classificador Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar o detector de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Variáveis para rastreamento de tempo
start_time = time.time()
interval = 3

# Criar a pasta para armazenar imagens
images_folder_path = r"C:\Users\eliak\visaocomputacional\imagens"
if not os.path.exists(images_folder_path):
    os.makedirs(images_folder_path)

# Lista de nomes de classes
class_names = ['user1', 'user2', 'user3']  # Adicione os nomes de usuários conforme necessário

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if time.time() - start_time >= interval:
            start_time = time.time()
            frame_to_recognize = frame[y:y+h, x:x+w]
            prediction = classify_frame(frame_to_recognize, model)
            class_id = np.argmax(prediction)
            class_name = class_names[class_id]
            confidence = prediction[0][class_id]

            print(f"Prediction: {class_name} ({confidence:.2f})")

            image_name = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
            image_path = os.path.join(images_folder_path, image_name)
            cv2.imwrite(image_path, frame_to_recognize)
            print("Image saved:", image_path)

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
