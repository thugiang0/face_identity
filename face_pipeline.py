import cv2
import yaml

from method.mtcnn_insightface import Recognition
from method.facenet.faceModuleTracker import faceDetectionRecognition

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class FacePipeline():
    def __init__(self, method, image):
        self.image = image
        self.method = method
        if method == "insightface":
            self.recognition = Recognition()
            targets, names = self.recognition.update_database(update=True)
        elif method == "facenet":
            self.recognition = faceDetectionRecognition()
        pass
    
    def recognize(self):
        return self.recognition.recognize_face(self.image)


img = "test/image/friends.jpg"
image = cv2.imread(img)
pipeline = FacePipeline(method="insightface", image=image)

result, recognized_image = pipeline.recognize()

cv2.imshow("face recognition", recognized_image)
cv2.waitKey(0)

