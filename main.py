from transformers import pipeline

classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
print(classifier("./download.jpeg")[0]["label"])
