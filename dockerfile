FROM python:3.8

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip

WORKDIR /media/giang/New Volume/FR/face_identity

COPY . .

COPY requirements_new.txt .
RUN pip install -r requirements_new.txt

RUN pip install python-multipart

CMD ["python", "api/main.py"]
