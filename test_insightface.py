import os
import cv2
from PIL import Image
from recognize import recognize_face, update_database
from mtcnn_insightface import Recognition
import json

img = "friends.jpg"

image = cv2.imread(img)

recognition = Recognition()

targets, names = recognition.update_database(update=True)

face_result, frame = recognition.recognize_face(image)
print(json.loads(face_result))

result_list = json.loads(face_result)

new_list_of_dicts = [{key: value for key, value in d.items() if key != "face"} for d in result_list]

cv2.imshow("face recognition", frame)
cv2.waitKey(0)

