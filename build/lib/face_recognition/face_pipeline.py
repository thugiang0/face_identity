import cv2
import yaml
import os

from .method.mtcnn_insightface import Recognition
from .method.facenet.faceModuleTracker import faceDetectionRecognition
from .utils.load_config import load_config

current_dir = os.path.dirname(__file__)

config_path = os.path.join(current_dir, "./configs/config.yaml")
config = load_config(config_path)

class FacePipeline():
    def __init__(self, method):
        self.method = method
        if method == "insightface":
            self.recognition = Recognition()
            targets, names = self.recognition.update_database(update=False)
        elif method == "facenet":
            self.recognition = faceDetectionRecognition()

        pass
      
    def recognize(self, image):
        try:
            return self.recognition.recognize_face(image)
        except AttributeError:
            print("The selected method does not support face recognition.")
            return None, image
    
    def add_face(self, name, img_path):
        return self.recognition.add_face(name, img_path)



