import os
import cv2
import shutil
import requests
import numpy as np
from glob import glob
from PIL import Image

import torch
from .detection import Detections
from method.mtcnn_model.mtcnn import MTCNN
from .inception_resnet_v1 import InceptionResnetV1
import yaml

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


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
    def __init__(self, person_dir='data_facenet/person', faces_dir='data_facenet/aligned', encode_dir=None, pretrained='vggface2', conf_thresh=config['face_recognition']['facenet']["threshold"]):
        self.person_dir = person_dir
        self.names = os.listdir(self.person_dir)
        self.faces_dir = faces_dir
        self.encode_dir = config["face_recognition"]["facenet"]["weight_path"]

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
            torch.save(encoding_dict, 'data_facenet/data.pt')
            print('Face Storage Created!')
            return encoding_dict
        else:
            try:
                encoding_dict = torch.load(self.encode_dir)
                print('Face Storage Loaded!')
                # encoding_dict.clear()
                print("encoding_dict: ", encoding_dict)
                # torch.save(encoding_dict, 'data.pt')

                return encoding_dict
            except:
                print('pt file has not valid content')

    def addFaces(self, name):
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

        encoding_dict = torch.load(self.encode_dir)
        encodes = []
        print(glob(f'{self.person_dir}/{name}/*'))
        for img_path in glob(f'{self.person_dir}/{name}/*'):
            save_name = img_path.split('/')[-1]
            encode, img_cropped = self.encoder(img_path, name, save_name)
            encodes.append(encode)
            mean_encode = torch.mean(torch.vstack(encodes), dim=0)
            encoding_dict[name] = mean_encode
        torch.save(encoding_dict, 'data_facenet/data.pt')
        print(encoding_dict)
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
                print(min_score[1])
                # with open('id_known.txt', 'a') as file:
                #     file.write(str(min_score[1]) + "\n")
                name = min(distances, key=lambda k: distances[k]) if min_score[1] < conf_thresh else 'Unknown'
                names.append(name)
            return names, crops
        
    def recognize_face(self, image):
        image = cv2.imread(image)
        fdr = faceDetectionRecognition(self.person_dir, self.faces_dir, self.encode_dir)
        encoding_dict = fdr.build_face_storage()
        # results = fdr.predict(img, encoding_dict)
        # names = results.display()
        outputs = self.face_detector.detect(image, landmarks=True)

        # recognized_image = results.show()
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

