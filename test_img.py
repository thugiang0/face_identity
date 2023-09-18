import os
import cv2
from PIL import Image
from recognize import recognize_face, update_database

img = "Aaron_Sorkin_0002.jpg"

image = cv2.imread(img)

update = True

targets, names = update_database(update)

face_result, frame = recognize_face(image, targets, names)

print(face_result)

cv2.imshow("face recognition", frame)
cv2.waitKey(0)

