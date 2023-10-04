from config import config

import cv2
from PIL import Image
from mtcnn_insightface import Recognition
from facenet.faceModuleTracker import faceDetectionRecognition


cfg = config()

recognize_method = cfg.method

print(recognize_method)

person_dir = 'face_database/facebank'
faces_dir = 'data_facenet/aligned'
encode_dir = 'data.pt'

img = "friends.jpg"

image = cv2.imread(img)

if recognize_method == "insightface":
    recognition = Recognition()
    
    targets, names = recognition.update_database(update=True)

    face_result, frame = recognition.recognize_face(image)

    cv2.imshow("face recognition", frame)
    cv2.waitKey(0)

elif recognize_method == "facenet":
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


