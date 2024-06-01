import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Carregar o modelo treinado
model_path = 'user_recognition_model_final.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Arquivo de modelo {model_path} não encontrado.")

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

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento Facial")
        self.root.geometry("800x600")

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.start_button = tk.Button(root, text="Iniciar Reconhecimento", command=self.start_recognition)
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="Parar Reconhecimento", command=self.stop_recognition)
        self.stop_button.pack()

        self.exit_button = tk.Button(root, text="Sair", command=self.exit_app)
        self.exit_button.pack()

        self.cap = None
        self.running = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def start_recognition(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop_recognition(self):
        if self.running:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()

    def exit_app(self):
        self.stop_recognition()
        self.root.destroy()

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = recognize_face(frame, self.face_cascade, model, class_names)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
