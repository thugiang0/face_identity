from  method.facenet.faceModuleTracker import faceDetectionRecognition
import os
import shutil

person_dir = 'data_facenet/person'
faces_dir = 'data_facenet/aligned'
encode_dir = 'data_facenet/data.pt'

new_person = "Monica"
new_face = f'{person_dir}/{new_person}'
os.mkdir(f'{person_dir}/{new_person}')

img = "Monica.jpg"
shutil.copy(img, new_face)

fdr = faceDetectionRecognition(person_dir, faces_dir, encode_dir)
fdr.addFaces(str(new_face))

def update_facenet(encode_dir, person_dir, faces_dir):
    fdr = faceDetectionRecognition(person_dir, faces_dir, encode_dir)

    fdr.addFaces(new_face, str(new_face))

    
