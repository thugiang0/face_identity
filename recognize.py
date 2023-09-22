import os
import cv2
from PIL import Image
import json
import numpy as np

from insightface.config import get_config
from mtcnn_model.mtcnn import MTCNN
from insightface.Learner import face_learner
from insightface.utils.prepare import load_facebank, draw_box_name, prepare_facebank
from mtcnn_model.utils.align_trans import get_reference_facial_points, warp_and_crop_face


cfg = get_config(False)

mtcnn = MTCNN()

learner = face_learner(cfg, True)

learner.threshold = 1.1066

if cfg.device.type == 'cpu':
    learner.load_state(cfg, 'cpu_final.pth', True, True)
else:
    learner.load_state(cfg, 'final.pth', True, True)
    
learner.model.eval()

print('learner loaded')

def detect_image(image):
    bboxes, scores, faces = mtcnn.detect(image, landmarks=True)
    return bboxes, scores, faces

def draw_face_box(image):
    bboxes, scores, faces = mtcnn.detect(image, landmarks=True)
    if bboxes is not None:
        for box in bboxes:
            bbox = list(map(int,box.tolist()))
            image = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    return image

# def recognize(image):

def update_database(update):
    if update:
        print("facebank update")
        targets, names = prepare_facebank(cfg, learner.model, mtcnn, tta=False)
    else:
        print("faceback loaded")
        targets, names = load_facebank(cfg)

    return targets, names
    
def recognize_face(frame, targets, names):

    result = []

    image = Image.fromarray(frame)

    bboxes, scores, landmarks = detect_image(frame)

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

        results, _ = learner.infer(cfg, faces, targets, tta=False)
        
        labels_box = []

        for idx, (bbox, score) in enumerate(zip(bboxes, scores)):

            name_id = names[results[idx] + 1]
            bbox = bbox.tolist()

            face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
       
            face_id = {
                "face": face.tolist(),
                "bbox": bbox,
                "score": score,
                "name": name_id
            }

            result.append(face_id)

            frame = draw_box_name(bbox, names[results[idx] + 1], frame)

            labels_box.append(names[results[idx] + 1])


        face_result = json.dumps(result, indent=4)
        # print(face_result)

        return face_result, frame



