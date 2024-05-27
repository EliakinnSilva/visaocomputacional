Documentação do Projeto de Detecção de Rostos e Mãos em Tempo Real
Objetivo
Este projeto tem como objetivo criar um sistema de detecção de rostos e mãos em tempo real usando uma webcam. O sistema detecta rostos na cena usando o classificador Haar Cascade e mãos usando o MediaPipe. O projeto também inclui um modelo de reconhecimento de objetos baseado no MobileNetV2.

Requisitos
Python 3.x
OpenCV (cv2)
NumPy
TensorFlow
MediaPipe
Arquivos do classificador Haar Cascade para detecção de rostos
Instalação
Certifique-se de ter o Python 3.x instalado no seu sistema.
Instale as dependências executando o seguinte comando:
bash
Copiar código
pip install opencv-python numpy tensorflow mediapipe
Baixe os arquivos do classificador Haar Cascade para detecção de rostos. Eles devem estar no mesmo diretório que o código.
Como usar
Execute o script app.py.
Uma janela de vídeo será aberta mostrando a saída da webcam.
A detecção de rostos e mãos será feita em tempo real e os resultados serão exibidos na janela de vídeo com tags indicando se é um rosto ou uma mão.
Detalhes do Código
O código utiliza a biblioteca OpenCV para capturar o vídeo da webcam, realizar a detecção de rostos e exibir os resultados em tempo real.
O classificador Haar Cascade é utilizado para a detecção de rostos na cena.
O MediaPipe é utilizado para a detecção de mãos na cena.
Um modelo de reconhecimento de objetos baseado no MobileNetV2 é incluído para reconhecimento adicional.
O código salva imagens de rostos detectados a cada 3 segundos.
Funções Principais
preprocess_image(image): Redimensiona e normaliza a imagem para o modelo.
create_model(num_classes): Cria e compila o modelo MobileNetV2 para reconhecimento de objetos.
classify_frame(frame, model): Classifica o frame usando o modelo treinado.
create_images_folder(folder_path): Cria uma pasta para armazenar as imagens se ela não existir.
main_loop(): Função principal que captura o vídeo, realiza a detecção de rostos e mãos, e exibe os resultados.
Exemplo de Uso
python
Copiar código
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
images_folder_path = r"C:\Users\eliak\visaocomputacional\imagens"
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
        # Adicione a tag "Face" ao lado do rosto detectado
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

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
            # Adicione a tag "Hand" ao lado da mão detectada
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
            cv2.putText(frame, "Hand", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exiba o frame
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

# Libere a captura de vídeo e feche todas as janelas
cap.release()
cv2.destroyAllWindows()
Limitações
A detecção de rostos e mãos pode não ser precisa em ambientes com baixa iluminação.
O desempenho pode ser afetado em sistemas com recursos limitados devido à carga computacional da detecção em tempo real.
Conclusão
Este projeto fornece uma implementação básica para detecção de rostos e mãos em tempo real usando OpenCV e MediaPipe, com a capacidade adicional de reconhecer objetos utilizando um modelo MobileNetV2. Ele pode ser expandido e melhorado conforme necessário para atender a requisitos específicos de aplicação.
