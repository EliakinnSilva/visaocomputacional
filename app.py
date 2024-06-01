import cv2
import os
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

def create_user_folder(folder_path, user_name):
    user_folder = os.path.join(folder_path, user_name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def capture_images(user_name, save_path, num_images=5, interval=2):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Erro", "Não foi possível acessar a câmera.")
        return

    count = 0
    user_folder = create_user_folder(save_path, user_name)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha na captura do frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            current_time = time.time()
            if current_time - start_time >= interval:
                start_time = current_time

                face_img = frame[y:y+h, x:x+w]
                img_name = f"{user_name}_{count}.jpg"
                img_path = os.path.join(user_folder, img_name)
                cv2.imwrite(img_path, face_img)
                print(f"Imagem {img_name} salva em {img_path}")
                count += 1
                
                if count >= num_images:
                    break

        cv2.imshow('Captura de Imagens', frame)

        if cv2.waitKey(1) % 256 == 27:  # Pressione ESC para sair
            break

        if count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()

class ImageCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Captura de Imagens")
        self.root.geometry("400x200")

        self.label = tk.Label(root, text="Escolha uma opção:")
        self.label.pack(pady=10)

        self.capture_button = tk.Button(root, text="Capturar imagens de um novo usuário", command=self.capture_images)
        self.capture_button.pack(pady=5)

        self.exit_button = tk.Button(root, text="Sair", command=root.quit)
        self.exit_button.pack(pady=5)

    def capture_images(self):
        user_name = simpledialog.askstring("Input", "Digite o nome do usuário:")
        if not user_name:
            messagebox.showerror("Erro", "Nome do usuário não pode ser vazio.")
            return

        try:
            num_images = int(simpledialog.askstring("Input", "Digite o número de imagens a serem capturadas:"))
        except ValueError:
            messagebox.showerror("Erro", "Número de imagens deve ser um inteiro.")
            return

        save_path = r"C:\Users\eliak\visaocomputacional\dataset"
        capture_images(user_name, save_path, num_images)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root)
    root.mainloop()
