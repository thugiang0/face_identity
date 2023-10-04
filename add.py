from facenet.faceModuleTracker import faceDetectionRecognition
import os
import shutil

person_dir = 'data_facenet/person'
faces_dir = 'data_facenet/aligned'
encode_dir = 'data.pt'

new_person = "Taylor_Swift"
new_face = f'{person_dir}/{new_person}'
os.mkdir(f'{person_dir}/{new_person}')

img = "Taylor_Swift.jpg"
shutil.copy(img, new_face)


def update_facenet(encode_dir, person_dir, faces_dir):
    fdr = faceDetectionRecognition(person_dir, faces_dir, encode_dir)

    fdr.addFaces(new_face, str(new_face))

    
