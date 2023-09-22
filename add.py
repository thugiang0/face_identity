from facenet.faceModuleTracker import faceDetectionRecognition
import os

person_dir = 'data_facenet/person'
faces_dir = 'data_facenet/aligned'
encode_dir = 'data.pt'

new_face = "Aaron_Sorkin"

fdr = faceDetectionRecognition(person_dir, faces_dir, encode_dir)

fdr.addFaces(new_face, str(new_face))


def update_facenet(encode_dir):
    fdr = faceDetectionRecognition(person_dir, faces_dir, encode_dir)

    fdr.addFaces(new_face, str(new_face))
