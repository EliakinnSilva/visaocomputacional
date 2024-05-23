import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import time
import mediapipe as mp
import os

# Pré-processamento de Imagens

def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))  # Redimensionamento para o tamanho esperado pelo modelo
    normalized_image = resized_image / 255.0  # Normalização dos valores de pixel
    return normalized_image

# Implementação do Modelo de Reconhecimento de Objetos

def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False)  # Carrega o modelo base sem a camada de classificação
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')  # Adiciona uma camada de classificação com softmax
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Desenvolvimento do Sistema de Classificação

def classify_frame(frame, model):
    preprocessed_frame = preprocess_image(frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Adiciona uma dimensão extra para a amostra única
    prediction = model.predict(preprocessed_frame)
    return prediction

# Criar pasta para armazenar as imagens se não existir
def create_images_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Carregamento do modelo
num_classes = 3  # Número de classes no conjunto de dados
model = create_model(num_classes)
# Carregue o modelo treinado...

# Carregar o classificador Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar o detector de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)  # 0 para a webcam padrão, ou troque por outro índice para usar uma webcam específica

# Variáveis para rastreamento de tempo
start_time = time.time()
interval = 3  # Intervalo de tempo em segundos para captura de frames

# Criar a pasta para armazenar imagens
images_folder_path = r"G:\AtvIngrid\imagens"
create_images_folder(images_folder_path)

while True:
    ret, frame = cap.read()  # Captura um frame da webcam

    if not ret:
        break

    # Converta o frame para escala de cinza para detecção de rostos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte rostos na cena
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Desenhe um retângulo ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Classifique o frame atual se o intervalo de tempo for atingido
        if time.time() - start_time >= interval:
            start_time = time.time()

            # Salve o frame atual para reconhecimento
            frame_to_recognize = frame[y:y+h, x:x+w]

            # Classifique o frame para reconhecimento
            prediction = classify_frame(frame_to_recognize, model)
            print("Prediction:", prediction)

            # Salve a imagem na pasta
            image_name = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
            image_path = os.path.join(images_folder_path, image_name)
            cv2.imwrite(image_path, frame_to_recognize)
            print("Image saved:", image_path)

    # Detecção de mãos
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenhe os pontos de referência das mãos no frame
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Exiba o frame
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

# Libere a captura de vídeo e feche todas as janelas
cap.release()
cv2.destroyAllWindows()
