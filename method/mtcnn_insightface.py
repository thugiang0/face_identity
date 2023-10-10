import os
import cv2
from PIL import Image
import json
import numpy as np
import yaml

from method.insightface.config import get_config
from method.mtcnn_model.mtcnn import MTCNN
from method.insightface.Learner import face_learner
from method.insightface.utils.prepare import load_facebank, draw_box_name, prepare_facebank
from method.mtcnn_model.utils.align_trans import get_reference_facial_points, warp_and_crop_face
from method.insightface.utils.prepare import add_facebank

import os
from pathlib import WindowsPath
import shutil


cfg = get_config(False)


mtcnn = MTCNN()


with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

learner = face_learner(cfg, True)
learner.threshold = config["face_recognition"]["insightface"]["threshold"]

learner.load_state(cfg, 'cpu_final.pth', True, True)

learner.model.eval()


class Recognition:

    def detect_image(self, image):
        bboxes, scores, landmarks = mtcnn.detect(image, landmarks=True)
        print("bbox: ", bboxes)
        return bboxes, scores, landmarks

    # def draw_face_box(self):
    #     bboxes, scores, faces = mtcnn.detect(image, landmarks=True)
    #     if bboxes is not None:
    #         for box in bboxes:
    #             bbox = list(map(int,box.tolist()))
    #             image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    #     return image

    def update_database(self, update):
        if update:
            print("facebank update")
            self.targets, self.names = prepare_facebank(cfg, learner.model, mtcnn, tta=False)
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

            results, _ = learner.infer(cfg, faces, self.targets, tta=False)
            
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
        path = f"face_database/facebank/{name}"
        os.mkdir(path)
        shutil.copy(img_path, path)
        path =  WindowsPath(path)
        add_facebank(cfg, learner.model, name, path, tta=False)



