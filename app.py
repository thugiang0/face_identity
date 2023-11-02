from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
from typing import List

app = FastAPI()

def get_image_shape(image):
    return image.size

@app.post("/upload/")
async def upload_image(files: List[UploadFile]):
    for file in files:
        if file.content_type.startswith('image'):
            image_data = await file.read()
            image = Image.open(BytesIO(image_data))
            image_shape = get_image_shape(image)
            return {"shape": image_shape}
        else:
            return {"error": "Invalid file type"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
