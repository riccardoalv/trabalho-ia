import cv2
import concurrent.futures
from transformers import pipeline
from PIL import Image

# Função para classificação de emoções em uma imagem
def classify_emotions(image):
    classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    return classifier(image)

def draw_square_with_label(img, x, y, w, h, label):
    # Desenhar um quadrado em torno do rosto
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Escrever a legenda em cima do quadrado
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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

# Carregar o modelo de detecção facial pré-treinado
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

emotions = ""

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Desenhar um quadrado em torno de cada rosto detectado com legenda
    for (x, y, w, h) in faces:
        draw_square_with_label(frame, x, y, w, h, "")

    key = cv2.waitKey(20)
    if key == 27: # Sair com ESC
        break
    # Processar o quadro em segundo plano
    executor.submit(process_frame_in_background, frame)

# Liberar recursos
vc.release()
cv2.destroyWindow("preview")

