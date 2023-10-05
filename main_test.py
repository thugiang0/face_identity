from configs.config import config

import cv2
from PIL import Image
from insightface.mtcnn_insightface import Recognition
from facenet.faceModuleTracker import faceDetectionRecognition
import yaml


with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


recognition_method = config['face_recognition']['method']

img = "test/image/friends.jpg"

image = cv2.imread(img)

if recognition_method == "insightface":
    recognition = Recognition()
    
    targets, names = recognition.update_database(update=True)

    face_result, frame = recognition.recognize_face(image)

    cv2.imshow("face recognition", frame)
    cv2.waitKey(0)

elif recognition_method == "facenet":
    person_dir = 'data_facenet/person'
    faces_dir = 'data_facenet/aligned'
    encode_dir = config["face_recognition"]["facenet"]["weight_path"]

    fdr = faceDetectionRecognition(person_dir, faces_dir, encode_dir)
    encoding_dict = fdr.build_face_storage()

    results = fdr.predict(img, encoding_dict)
    results.save()
    name = results.display()
    print("name: ", name)

    cv2.imshow('Friends', results.show())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    cv2.imshow("face recognition", image)
    cv2.waitKey(0)


