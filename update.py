from insightface.config import get_config
from mtcnn_model.mtcnn import MTCNN
from insightface.Learner import face_learner

from insightface.utils.prepare import add_facebank

import os
from pathlib import WindowsPath


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


def add_to_database(img_path):
    path =  WindowsPath(img_path)
    add_facebank(cfg, learner.model, path, tta=False)

