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

def main():
    save_path = r"C:\Users\eliak\visaocomputacional\dataset"

    root = tk.Tk()
    root.withdraw()  # Esconder a janela principal do Tkinter

    while True:
        choice = simpledialog.askstring("Menu", "Escolha uma opção:\n1. Capturar imagens de um novo usuário\n2. Sair")
        
        if choice == '1':
            user_name = simpledialog.askstring("Input", "Digite o nome do usuário:")
            if not user_name:
                messagebox.showerror("Erro", "Nome do usuário não pode ser vazio.")
                continue

            try:
                num_images = int(simpledialog.askstring("Input", "Digite o número de imagens a serem capturadas:"))
            except ValueError:
                messagebox.showerror("Erro", "Número de imagens deve ser um inteiro.")
                continue

            capture_images(user_name, save_path, num_images)
        elif choice == '2':
            print("Encerrando o programa.")
            break
        else:
            messagebox.showerror("Erro", "Opção inválida. Por favor, escolha novamente.")

if __name__ == "__main__":
    main()
