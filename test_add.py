from face_pipeline import FacePipeline

pipeline = FacePipeline(method="insightface")

pipeline.add_face(name="Phoebe", img_path="Phoebe.jpg")