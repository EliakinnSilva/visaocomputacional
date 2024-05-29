import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import time
import mediapipe as mp
import os
import pickle

def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    return normalized_image

def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_database_images(database_path):
    database = {}
    for user_name in os.listdir(database_path):
        user_folder = os.path.join(database_path, user_name)
        if os.path.isdir(user_folder):
            user_images = []
            for image_name in os.listdir(user_folder):
                image_path = os.path.join(user_folder, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    user_images.append(image)
            database[user_name] = user_images
    return database

def extract_embeddings(images, model):
    embeddings = []
    for image in images:
        preprocessed_image = preprocess_image(image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        embedding = model.predict(preprocessed_image)
        embeddings.append(embedding)
    return np.mean(embeddings, axis=0)

def recognize_face(face_image, embeddings_db, model, threshold=0.6):
    preprocessed_face = preprocess_image(face_image)
    preprocessed_face = np.expand_dims(preprocessed_face, axis=0)
    face_embedding = model.predict(preprocessed_face)
    
    min_dist = float('inf')
    identity = None

    for name, db_embedding in embeddings_db.items():
        dist = np.linalg.norm(face_embedding - db_embedding)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist < threshold:
        return identity
    else:
        return "Unknown"

def create_images_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Carregar imagens e extrair embeddings do banco de dados
database_path = r"C:\Users\eliak\visaocomputacional\dataset"
database_images = load_database_images(database_path)
embeddings_db = {name: extract_embeddings(images, model) for name, images in database_images.items()}

num_classes = len(embeddings_db)
model = create_model(num_classes)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

start_time = time.time()
interval = 3

images_folder_path = r"C:\Users\eliak\visaocomputacional\imagens"
create_images_folder(images_folder_path)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Falha na captura do frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Faces detectadas: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if time.time() - start_time >= interval:
            start_time = time.time()
            frame_to_recognize = frame[y:y+h, x:x+w]
            identity = recognize_face(frame_to_recognize, embeddings_db, model)
            print(f"Identity: {identity}")

            image_name = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
            image_path = os.path.join(images_folder_path, image_name)
            cv2.imwrite(image_path, frame_to_recognize)
            print(f"Image saved: {image_path}")

            cv2.putText(frame, identity, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
            cv2.putText(frame, "Hand", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
