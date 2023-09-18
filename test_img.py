import os
import cv2
from PIL import Image
from recognize import recognize_face

img = "Aaron_Sorkin_0002.jpg"

image = cv2.imread(img)

face_result, frame = recognize_face(image)

print(face_result)

cv2.imshow("face recognition", frame)
cv2.waitKey(0)

