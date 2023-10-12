from fastapi import FastAPI, Request, File, UploadFile, Form
from typing import Optional, List
from face_recognition.face_pipeline import FacePipeline
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import os
import json
import shutil
import tempfile

app = FastAPI()

@app.post("/recognize")
async def recognize(files: List[UploadFile]):
    results = []
    for file in files:
        img_request = await file.read()
        img = cv2.imdecode(np.frombuffer(img_request, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        pipeline = FacePipeline(method="insightface")
        result, image_result = pipeline.recognize(img)
    
        result_list = json.loads(result)

        result_dict = [{key: value for key, value in d.items() if key != "face"} for d in result_list]

        results.append(result_dict)

    return results


@app.post("/update_data")
async def update_data(file: UploadFile, name: str = Form(...)):
    img_request = await file.read()
    img = cv2.imdecode(np.frombuffer(img_request, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    upload_folder = "api/upload_img/"
    image_path = os.path.join(upload_folder, file.filename)
    cv2.imwrite(image_path, img)

    pipeline = FacePipeline(method="insightface")
    pipeline.add_face(name=name, img_path=image_path)

    return {
        "status": "updated"
        }
