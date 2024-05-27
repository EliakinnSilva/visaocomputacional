import tkinter as tk
from PIL import Image, ImageTk
import cv2
import dlib

class Application:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Detecção e Classificação em Tempo Real")
        
        # Inicialização da captura de vídeo
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
        
        # Inicialização da interface gráfica
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        # Botão para iniciar o processamento
        self.btn_start = tk.Button(window, text="Iniciar Processamento", width=20, command=self.start_processing)
        self.btn_start.pack(pady=10)
        
        # Atualização do frame
        self.delay = 10
        self.update()

        # Inicialização do detector de rostos
        self.face_detector = dlib.get_frontal_face_detector()

        # Inicialização do detector de mãos (você pode substituir isso por outro método)
        self.hand_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')

    def start_processing(self):
        while True:
            ret, frame = self.vid.read()
            if ret:
                # Conversão para escala de cinza para detecção de rostos e mãos
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detecção de rostos
                faces = self.face_detector(gray)
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Detecção de mãos
                hands = self.hand_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (hx, hy, hw, hh) in hands:
                    cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (255, 0, 0), 2)

                # Exibição do frame com as detecções
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Libere a captura de vídeo e feche a janela quando terminar
        self.vid.release()
        cv2.destroyAllWindows()

    def update(self):
        # Função para atualizar o frame exibido na interface gráfica
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

def main():
    root = tk.Tk()
    app = Application(root)
    root.mainloop()

if __name__ == "__main__":
    main()
