from face_recognition.face_pipeline import FacePipeline

pipeline = FacePipeline(method="insightface")

pipeline.add_face(name="Jennie", img_path="test/image/jennie.jpg")