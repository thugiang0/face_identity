import cv2
import yaml

from method.mtcnn_insightface import Recognition
from method.facenet.faceModuleTracker import faceDetectionRecognition
from utils.load_config import load_config

config_path = "configs/config.yaml"
config = load_config(config_path)

class FacePipeline():
    def __init__(self, method):
        self.method = method
        if method == "insightface":
            self.recognition = Recognition()
            targets, names = self.recognition.update_database(update=True)
        elif method == "facenet":
            self.recognition = faceDetectionRecognition()
        pass
      
    def recognize(self, image):
        return self.recognition.recognize_face(image)
    
    def add_face(self, name, img_path):
        return self.recognition.add_face(name, img_path)



