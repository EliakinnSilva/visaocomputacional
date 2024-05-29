# Projeto de Captura de Imagens de Rostos

## Objetivo
Este projeto tem como objetivo capturar imagens do rosto de um usuário utilizando uma webcam e salvar essas imagens em um diretório específico.

## Requisitos
- Python 3.x
- OpenCV (cv2)
- Tkinter (para a interface de usuário)
- Arquivo do classificador Haar Cascade para detecção de rostos

## Instalação
1. Certifique-se de ter o Python 3.x instalado no seu sistema.
2. Instale as dependências executando o seguinte comando:
==> pip install opencv-python-headless opencv-python-headless <==
3. Baixe o arquivo do classificador Haar Cascade para detecção de rostos (haarcascade_frontalface_default.xml).

## Como Usar
1. Execute o script capture_face_images.py.
2. Uma janela será aberta solicitando o nome do usuário e o número de imagens a serem capturadas.
3. Posicione seu rosto dentro da área visível da webcam.
4. As imagens do rosto serão capturadas automaticamente de acordo com o número especificado.
5. As imagens serão salvas em um diretório específico.

## Detalhes do Código
O código utiliza a biblioteca OpenCV para capturar o vídeo da webcam e detectar rostos na cena. As imagens do rosto são capturadas automaticamente e salvas em um diretório específico. O script também utiliza a biblioteca Tkinter para uma interface de usuário básica.

## Limitações
- A detecção de rostos pode não ser tão precisa em todas as condições de iluminação e posição da câmera.
- O desempenho pode variar dependendo da velocidade do processador e da webcam utilizada.

- 
