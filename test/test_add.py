from face_recognition.face_pipeline import FacePipeline

pipeline = FacePipeline(method="insightface")

pipeline.add_face(name="Taylor_Swift", img_path="test/image/Taylor_Swift.jpg")