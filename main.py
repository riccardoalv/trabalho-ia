import cv2
import concurrent.futures
from transformers import pipeline
from PIL import Image

# Função para classificação de emoções em uma imagem
def classify_emotions(image):
    classifier = pipeline("image-classification", model="jayanta/google-vit-base-patch16-224-cartoon-emotion-detection")
    return classifier(image)

# Função para processamento de cada frame em segundo plano
def process_frame_in_background(frame):
    # Converter frames para imagens Pillow
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Classificar emoções na imagem
    emotions = classify_emotions(image)
    print(emotions)

# Inicializar a captura de vídeo
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # Tentar obter o primeiro frame
    rval, frame = vc.read()
else:
    rval = False

# Iniciar o pool de execução para processar quadros em segundo plano
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # Sair com ESC
        break
    # Processar o quadro em segundo plano
    executor.submit(process_frame_in_background, frame)

# Liberar recursos
vc.release()
cv2.destroyWindow("preview")

