import cv2
import os
import time
import face_recognition
import tkinter as tk
from tkinter import simpledialog, messagebox

def create_user_folder(folder_path, user_name):
    user_folder = os.path.join(folder_path, user_name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def capture_images(user_name, save_path, num_images=5, interval=2):
    cap = cv2.VideoCapture(0)
    count = 0
    
    user_folder = create_user_folder(save_path, user_name)
    
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha na captura do frame.")
            break

        current_time = time.time()
        if current_time - start_time >= interval:
            start_time = current_time

            # Detectar rostos na imagem
            face_locations = face_recognition.face_locations(frame)

            for face_location in face_locations:
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

                face_img = frame[top:bottom, left:right]
                img_name = f"{user_name}_{count}.jpg"
                img_path = os.path.join(user_folder, img_name)
                cv2.imwrite(img_path, face_img)
                print(f"Imagem {img_name} salva em {img_path}")
                count += 1

                if count >= num_images:
                    break

        cv2.imshow('Captura de Imagens', frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # Pressione ESC para sair
            break

        if count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()

def load_known_faces(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for user_folder in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_folder)
        if not os.path.isdir(user_path):
            continue

        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(img)

            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(user_folder)

    return known_face_encodings, known_face_names

def recognize_user(frame, known_face_encodings, known_face_names):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        return name
    return None

def main():
    save_path = r"C:\Users\eliak\visaocomputacional\dataset"
    known_face_encodings, known_face_names = load_known_faces(save_path)

    root = tk.Tk()
    root.withdraw()  # Esconder a janela principal do Tkinter

    while True:
        choice = simpledialog.askstring("Menu", "Escolha uma opção:\n1. Capturar imagens de um novo usuário\n2. Reconhecer usuário\n3. Sair")

        if choice == '1':
            user_name = simpledialog.askstring("Input", "Digite o nome do usuário:")
            num_images = int(simpledialog.askstring("Input", "Digite o número de imagens a serem capturadas:"))
            capture_images(user_name, save_path, num_images)
            known_face_encodings, known_face_names = load_known_faces(save_path)
        elif choice == '2':
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Falha na captura do frame.")
                    break

                name = recognize_user(frame, known_face_encodings, known_face_names)
                if name:
                    cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('Reconhecimento de Usuário', frame)

                k = cv2.waitKey(1)
                if k % 256 == 27:  # Pressione ESC para sair
                    break

            cap.release()
            cv2.destroyAllWindows()
        elif choice == '3':
            print("Encerrando o programa.")
            break
        else:
            messagebox.showerror("Erro", "Opção inválida. Por favor, escolha novamente.")

if __name__ == "__main__":
    main()
