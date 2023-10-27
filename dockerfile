FROM python:3.8

WORKDIR /media/giang/New Volume/FR/face_identity

COPY . .

COPY requirements_new.txt .
RUN pip install -r requirements_new.txt

CMD ["uvicorn", "api.main:app", "--reload"]
