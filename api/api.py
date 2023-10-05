from fastapi import FastAPI, Request, File, UploadFile, Form
from typing import Optional, List
from mtcnn_insightface import update_database, recognize_face
from update import add_to_database
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import os
import json

app = FastAPI()

@app.post("/recognize")
async def recognize(files: List[UploadFile]):
    targets, names = update_database(False)
    results = []
    for file in files:
        img_request = await file.read()
        img = cv2.imdecode(np.frombuffer(img_request, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        result, image_result = recognize_face(img, targets, names)
    
        result_list = json.loads(result)

        result_dict = [{key: value for key, value in d.items() if key != "face"} for d in result_list]

        results.append(result_dict)

    return results


@app.post("/update_data")
async def update_data(file: UploadFile, name: str = Form(...)):
    img_request = await file.read()
    img = cv2.imdecode(np.frombuffer(img_request, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    save_database = "face_database/facebank/" + name
    os.mkdir(save_database)
    img_path = os.path.join(save_database, file.filename)
    cv2.imwrite(img_path, img)
    add_to_database(save_database)
    return {
        "status": "updated"
        }