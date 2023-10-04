import cv2
from facenet.faceModuleTracker import faceDetectionRecognition


person_dir = 'data_facenet/person'
faces_dir = 'data_facenet/aligned'
encode_dir = 'data.pt'
img = 'Taylor_Swift.jpg'

fdr = faceDetectionRecognition(person_dir, faces_dir, encode_dir)
encoding_dict = fdr.build_face_storage()

results = fdr.predict(img, encoding_dict)
results.save()
name = results.display()
print("name: ", name)

cv2.imshow('Friends', results.show())
cv2.waitKey(0)
cv2.destroyAllWindows()

