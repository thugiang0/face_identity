import os
import cv2
from PIL import Image
import json
import numpy as np
import yaml

from .insightface.config import get_config
from .mtcnn_model.mtcnn import MTCNN
from .insightface.Learner import face_learner
from .insightface.utils.prepare import load_facebank, draw_box_name, prepare_facebank
from .mtcnn_model.utils.align_trans import get_reference_facial_points, warp_and_crop_face
from .insightface.utils.prepare import add_facebank
from ..utils.load_config import load_config

import os
from pathlib import WindowsPath
import shutil


cfg = get_config(False)

current_dir = os.path.dirname(__file__)

config_path = os.path.join(current_dir, "../configs/config.yaml")
config = load_config(config_path)


class Recognition:

    def __init__(self):
        self.mtcnn = MTCNN()
        learner = face_learner(cfg, True)
        self.learner = learner
        self.learner.threshold = config["face_recognition"]["insightface"]["threshold"]

        self.learner.load_state(cfg, 'cpu_final.pth', True, True)

        self.learner.model.eval()

    def detect_image(self, image):
        bboxes, scores, landmarks = self.mtcnn.detect(image, landmarks=True)

        return bboxes, scores, landmarks

    def update_database(self, update):
        if update:
            print("facebank update")
            self.targets, self.names = prepare_facebank(cfg, self.learner.model, self.mtcnn, tta=False)
        else:
            print("faceback loaded")
            self.targets, self.names = load_facebank(cfg)

        return self.targets, self.names
        
    def recognize_face(self, frame):

        result = []

        image = Image.fromarray(frame)

        bboxes, scores, landmarks = self.detect_image(frame)

        if bboxes is None:
            s = "No detection"
            return s, frame
        
        else:

            bboxes = bboxes.astype(np.float64)
            bboxes = bboxes.astype(int)

            landmarks = np.hstack((landmarks[:, :, 0], landmarks[:, :, 1]))

            faces = []
            refrence = get_reference_facial_points(default_square= True)
            for landmark in landmarks:
                facial5points = [[landmark[j],landmark[j+5]] for j in range(5)]
                warped_face = warp_and_crop_face(np.array(image), facial5points, refrence, crop_size=(112,112))
                faces.append(Image.fromarray(warped_face))

            results, _ = self.learner.infer(cfg, faces, self.targets, tta=False)
            
            labels_box = []

            for idx, (bbox, score) in enumerate(zip(bboxes, scores)):

                name_id = self.names[results[idx] + 1]
                bbox = bbox.tolist()

                face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
                face_id = {
                    "face": face.tolist(),
                    "bbox": bbox,
                    "score": score,
                    "name": name_id
                }

                result.append(face_id)

                frame = draw_box_name(bbox, self.names[results[idx] + 1], frame)

                labels_box.append(self.names[results[idx] + 1])


            face_result = json.dumps(result, indent=4)
            # print(face_result)

            return face_result, frame
        
    def add_face(self, name, img_path):
        path = os.path.join(os.path.dirname(__file__), f"../face_database/facebank/{name}")
        os.mkdir(path)
        shutil.copy(img_path, path)
        path =  WindowsPath(path)
        add_facebank(cfg, self.learner.model, name, path, tta=False)



