import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

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

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)  # 0 para a webcam padrão, ou troque por outro índice para usar uma webcam específica

# Criação e carregamento do modelo
num_classes = 3  # Número de classes no conjunto de dados
model = create_model(num_classes)
# Carregue o modelo treinado...

while True:
    ret, frame = cap.read()  # Captura um frame da webcam

    if not ret:
        break

    # Classifica o frame atual
    prediction = classify_frame(frame, model)

    # Exiba o frame e os resultados da classificação
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

# Libere a captura de vídeo e feche todas as janelas
cap.release()
cv2.destroyAllWindows()
