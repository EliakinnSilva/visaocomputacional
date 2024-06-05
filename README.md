# Projeto de Captura de Imagens de Rostos

## Objetivo
Este projeto tem como objetivo capturar imagens do rosto de um usuário utilizando uma webcam e salvar essas imagens em um diretório específico.

## Detalhamentos Técnicos

- **Linguagem:** Python
- **Compilador/Interpretador:** Não aplicável, Python é uma linguagem interpretada.
- **Fontes de Pesquisa:** Não especificadas.
- **Origem do Código Inicial:** Desenvolvido internamente.
- **Treinamento da IA:** Utilizou-se transfer learning com a MobileNetV2, uma rede neural pré-treinada no conjunto de dados ImageNet. O treinamento foi feito utilizando o conjunto de dados capturados pelo próprio sistema, utilizando técnicas de aumento de dados como rotação, mudança de largura e altura, cisalhamento, zoom e inversão horizontal. O modelo foi compilado com otimizador Adam e função de perda de entropia cruzada categórica. O treinamento foi interrompido precocemente se não houve melhoria na perda de validação após um número específico de épocas.

## Requisitos
- Python 3.x
- OpenCV (cv2)
- Tkinter (para a interface de usuário)
- Arquivo do classificador Haar Cascade para detecção de rostos

## Instalação
1. Certifique-se de ter o Python 3.x instalado no seu sistema.
2. Instale as dependências executando o seguinte comando:

```bash
pip install opencv-python-headless opencv-python-headless
