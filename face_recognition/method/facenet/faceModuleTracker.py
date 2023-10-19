import os
import cv2
import shutil
import requests
import numpy as np
from glob import glob
from PIL import Image

import torch
from .detection import Detections
from ..mtcnn_model.mtcnn import MTCNN
from .inception_resnet_v1 import InceptionResnetV1
import yaml

from ...utils.load_config import load_config

current_dir = os.path.dirname(__file__)

config_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "configs/config.yaml")
config = load_config(config_path)


class faceDetectionRecognition:
    """
    face Detection and Recognition Module.

    this class by using facenet-pytorch package first detects and encodes faces who wanted to recognize
    then save encodes in .pt file to future usage (every time you can add new face to data storage of faces)

    After creation of encoded file prediction based on comparison of new image encode and encodes is done

    Keyword Arguments:
        :param person_dir: {str} -> directory of persons who is wanted to recognize (every person has one or more image in a directory)
        :param faces_dir: {str} -> a directory to save aligned images
        :param encode_dir: {str} -> a directory to save or load face encoded data
        :param pretrained: {str} -> 'vggface2' 107Mb or 'casia-webface' 111Mb
    """
    def __init__(self, pretrained='vggface2', conf_thresh=config['face_recognition']['facenet']["threshold"]):
        # self.person_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data_facenet/person')
        # self.names = os.listdir(self.person_dir)
        self.faces_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data_facenet/aligned')
        self.encode_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data_facenet/data.pt")

        self.face_detector = MTCNN(image_size=160, margin=0.1, thresholds=[0.6, 0.7, 0.85], keep_all=True)
        self.face_encoder = InceptionResnetV1(pretrained=pretrained).eval()
        self.conf_thresh = conf_thresh

    def build_face_storage(self):
        """
        encode persons image and save it to data.pt file

        :return: {dict}
        a dictionary of person names and mean encode of each person {person_name:encode}
        """
        if self.encode_dir is None:
            print("person_dir ", self.person_dir)
            encoding_dict = {}
            for name in os.listdir(self.person_dir):
                encodes = []
                # images of one person
                for img_path in glob(f'{self.person_dir}/{name}/*'):
               
                    # save_name for aligned image
                    save_name = img_path.split('/')[-1]
                    encode, img_cropped = self.encoder(img_path, name, save_name)
                    encodes.append(encode)
                    # mean of encodes for one person
                    mean_encode = torch.mean(torch.vstack(encodes), dim=0)
                    encoding_dict[name] = mean_encode

            # saving all of encodes
            torch.save(encoding_dict, self.encode_dir)
            print('Face Storage Created!')
            return encoding_dict
        else:
            try:
                encoding_dict = torch.load(self.encode_dir)
                print('Face Storage Loaded!')
                # encoding_dict.clear()
                # torch.save(encoding_dict, 'data.pt')

                return encoding_dict
            except:
                print('pt file has not valid content')

    def addFaces(self, name, img_path):
        """
        adding new face encode to encodes
        :param path: {str} -> path of a directory contains new face images
        :param name: {str} -> name of new face
        :return: None
        """
        # if name not in self.names:
            # create a directory for new person and copy images to it
            # os.mkdir(f'{self.person_dir}/{name}')
        # for img_name in os.listdir(path):
        #     src = os.path.join(path, img_name)
        #     dst = os.path.join(self.person_dir, name, img_name)
        #     shutil.copy(src, dst)
        print("add name: ", name)
        encoding_dict = torch.load(self.encode_dir)
        encodes = []
        # print(new_face)
        # for img_path in os.listdir(new_face):
            # img_path = os.path.join(new_face, img_path)
        save_name = img_path.split('/')[-1]
        encode, img_cropped = self.encoder(img_path, name, save_name)
        encodes.append(encode)
        mean_encode = torch.mean(torch.vstack(encodes), dim=0)
        encoding_dict[name] = mean_encode
        torch.save(encoding_dict, self.encode_dir)
        print("save name: ", save_name)
        print(f"The {name}'s face added!")
        # else:
        #     print(f"The {name}'s face exists!")


    def compare(self, img, encoding_dict):
        """
        comparison of new image encode and encoding_dict and choose one person
        if it is close
        :param img: {Image.Image} image to comparison
        :param encoding_dict: {dict} a dictionary of names and encodings
        :param conf_thresh: a threshold to separate known and unknown face
        :return:
            predicted name
            cropped face of predicted name
        """
        crops = self.face_detector(img)
        conf_thresh = self.conf_thresh
  
        if crops is not None:
            self.face_encoder.classify = True
            encodes = self.face_encoder(crops).detach()
            names = []
            
            for i in range(len(encodes)):
                encode = encodes[i]
                distances = {}
                for name, embed in encoding_dict.items():
                    # comparison
                    dist = torch.dist(encode, embed).item()
                    distances[name] = dist
                # min of distance if less than conf_thresh
                min_score = min(distances.items(), key=lambda x: x[1])
                # with open('id_known.txt', 'a') as file:
                #     file.write(str(min_score[1]) + "\n")
                name = min(distances, key=lambda k: distances[k]) if min_score[1] < conf_thresh else 'Unknown'
                names.append(name)
            return names, crops
        
    def recognize_face(self, image):
    
        encoding_dict = self.build_face_storage()
   
        outputs = self.face_detector.detect(image, landmarks=True)

        results_dict = []
        names, crops = self.compare(image, encoding_dict)
        for i, (box, score) in enumerate(zip(outputs[0], outputs[1])):
         
            x1, y1, x2, y2 = list(map(lambda x: int(x), box))
            scale = round(((x2 - x1) + 78) / 75)
            (w, h), _ = cv2.getTextSize(names[i], cv2.FONT_HERSHEY_PLAIN, scale, 2)
            cv2.rectangle(image, (x1, y2 + h + 2), (x1 + w, y2), (255, 0, 0), -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=5)
            cv2.putText(image, names[i], (x1, y2 + h), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), 2)

            # if landmarks == True:
            for point in outputs[2][i]:
                x, y = int(point[0]), int(point[1])
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

            result = {
                # "face": image.tolist(),
                "bbox": box,
                "score": score,
                "name": names[i]
            }
            results_dict.append(result)

        return results_dict, image
    
    
    def encoder(self, img_path, name, save_name):
        """
        encoding one image and save aligned face

        :param img_path: {str}
        :param name: {str} -> face name
        :param save_name: {str}
        :return:
        """
        img = Image.open(img_path)
        img_cropped = self.face_detector(img, save_path=f'{self.faces_dir}/{name}/{save_name}')
        try:
            self.face_encoder.classify = True
            encode = self.face_encoder(img_cropped).detach()
            return encode, img_cropped
        except ValueError:
            print('No Face detected in one of storage Faces; change or remove image from storage')

    def add_face(self, name, img_path):
        # new_face = f'{self.person_dir}/{name}'
        # # os.mkdir(new_face)
        # # shutil.copy(img_path, new_face)
        self.addFaces(name, img_path)

    

