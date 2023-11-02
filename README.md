# Face Identity

## Detection
- MTCNN

## Recognition
- Insightface
- Facenet

## How to use
- Clone
    ```
    git clone https://github.com/thugiang0/face_identity.git
    ```

- Install
     ```
    pip -r install requirements.txt
    ```

- Test
    - Recognize
    ```
    python test/test_recognition.py
    ```

    - Add face to face database
    ```
    python test/test_add.py
    ```

- FastAPI
    ```
    uvicorn main:app --reload
    ```

- Demo 
    ```
    streamlit run demo/app.py
    ```


    
