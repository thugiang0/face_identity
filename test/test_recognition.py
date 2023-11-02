from face_recognition.face_pipeline import FacePipeline
import cv2
import torch

if __name__ == '__main__':

    img = "test/image/blackpink.jpg"
    image = cv2.imread(img)
    pipeline = FacePipeline(method="insightface")

    result, recognized_image = pipeline.recognize(image)
    print(result)
    cv2.imwrite("img.jpg", recognized_image)

    # cv2.imshow("face recognition", recognized_image)
    # cv2.waitKey(0)
