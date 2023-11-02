
from starlette.datastructures import UploadFile


file_path = "img.jpg"

with open(file_path, "rb") as file:
    upload_file = UploadFile(file)
    print(type(upload_file))