from face_pipeline import FacePipeline
import cv2

img = "test/image/friends.jpg"
image = cv2.imread(img)
pipeline = FacePipeline(method="insightface")

result, recognized_image = pipeline.recognize(image)

cv2.imshow("face recognition", recognized_image)
cv2.waitKey(0)