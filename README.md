<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentação do Projeto</title>
</head>
<body>
    <h1>Documentação do Projeto de Detecção de Rostos e Objetos em Tempo Real</h1>

    <h2>Objetivo</h2>
    <p>Este projeto tem como objetivo criar um sistema de detecção de rostos e objetos em tempo real usando uma webcam. O sistema detecta rostos na cena usando o classificador Haar Cascade e objetos usando o modelo YOLOv3.</p>

    <h2>Requisitos</h2>
    <ul>
        <li>Python 3.x</li>
        <li>OpenCV (cv2)</li>
        <li>NumPy</li>
        <li>Arquivos do modelo YOLOv3 (<code>yolov3.weights</code>, <code>yolov3.cfg</code> e <code>coco.names</code>)</li>
        <li>Arquivos do classificador Haar Cascade para detecção de rostos</li>
    </ul>

    <h2>Instalação</h2>
    <ol>
        <li>Certifique-se de ter o Python 3.x instalado no seu sistema.</li>
        <li>Instale as dependências executando o seguinte comando:
            <pre>pip install opencv-python numpy</pre>
        </li>
        <li>Baixe os arquivos do modelo YOLOv3 (<code>yolov3.weights</code>, <code>yolov3.cfg</code> e <code>coco.names</code>) e os arquivos do classificador Haar Cascade para detecção de rostos. Eles devem estar no mesmo diretório que o código.</li>
    </ol>

    <h2>Como usar</h2>
    <ol>
        <li>Execute o script <code>detect_faces_and_objects.py</code>.</li>
        <li>Uma janela de vídeo será aberta mostrando a saída da webcam.</li>
        <li>A detecção de rostos e objetos será feita em tempo real e os resultados serão exibidos na janela de vídeo.</li>
    </ol>

    <h2>Detalhes do Código</h2>
    <ul>
        <li>O código utiliza a biblioteca OpenCV para capturar o vídeo da webcam, realizar a detecção de rostos e objetos, e exibir os resultados em tempo real.</li>
        <li>O classificador Haar Cascade é utilizado para a detecção de rostos na cena.</li>
        <li>O modelo YOLOv3 é utilizado para a detecção de objetos na cena.</li>
        <li>Os arquivos do modelo YOLOv3 devem estar presentes no mesmo diretório que o código para que a detecção de objetos funcione corretamente.</li>
    </ul>

    <h2>Limitações</h2>
    <ul>
        <li>A detecção de objetos pode não ser tão precisa em ambientes com baixa iluminação ou com objetos pequenos.</li>
        <li>O desempenho pode ser afetado em sistemas com recursos limitados devido à carga computacional da detecção em tempo real.</li>
    </ul>
</body>
</html>
