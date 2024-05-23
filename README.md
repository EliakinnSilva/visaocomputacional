Documentação do Projeto de Detecção de Rostos e Objetos em Tempo Real
Objetivo
Este projeto tem como objetivo criar um sistema de detecção de rostos e objetos em tempo real usando uma webcam. O sistema detecta rostos na cena usando o classificador Haar Cascade e objetos usando o modelo YOLOv3.

Requisitos
Python 3.x
OpenCV (cv2)
NumPy
Arquivos do modelo YOLOv3 (yolov3.weights, yolov3.cfg e coco.names)
Arquivos do classificador Haar Cascade para detecção de rostos
Instalação
Certifique-se de ter o Python 3.x instalado no seu sistema.
Instale as dependências executando o seguinte comando:
Copiar código
pip install opencv-python numpy
Baixe os arquivos do modelo YOLOv3 (yolov3.weights, yolov3.cfg e coco.names) e os arquivos do classificador Haar Cascade para detecção de rostos. Eles devem estar no mesmo diretório que o código.
Como usar
Execute o script detect_faces_and_objects.py.
Uma janela de vídeo será aberta mostrando a saída da webcam.
A detecção de rostos e objetos será feita em tempo real e os resultados serão exibidos na janela de vídeo.
Detalhes do Código
O código utiliza a biblioteca OpenCV para capturar o vídeo da webcam, realizar a detecção de rostos e objetos, e exibir os resultados em tempo real.
O classificador Haar Cascade é utilizado para a detecção de rostos na cena.
O modelo YOLOv3 é utilizado para a detecção de objetos na cena.
Os arquivos do modelo YOLOv3 devem estar presentes no mesmo diretório que o código para que a detecção de objetos funcione corretamente.
Limitações
A detecção de objetos pode não ser tão precisa em ambientes com baixa iluminação ou com objetos pequenos.
O desempenho pode ser afetado em sistemas com recursos limitados devido à carga computacional da detecção em tempo real.
Esta documentação fornece uma visão geral do projeto, incluindo requisitos, instruções de instalação, como usar o sistema e detalhes sobre o código implementado.
