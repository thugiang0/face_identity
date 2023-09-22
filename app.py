import streamlit as st
import cv2
import numpy as np
from PIL import Image
from recognize import draw_face_box, update_database, recognize_face
import os
from update import add_to_database
import json



default_img = "default.jpg"

st.title("Face Recognition")

st.sidebar.title("Menu")
st.sidebar.subheader("Detection")

st.sidebar.subheader("Take picture")


st.sidebar.markdown("----")

# Add face to database

img_file_add = st.sidebar.file_uploader("Add to database", type=["jpg", "png", "jpeg"])

if img_file_add is not None:
    image_add = np.array(Image.open(img_file_add))
else:
    image_add = np.array(Image.open(default_img))

st.sidebar.text("add image")
st.sidebar.image(image_add)

name_person = st.sidebar.text_input("Input name")

if st.sidebar.button("Update"):
    if img_file_add is not None and name_person:
        

        save_folder = os.path.join("face_database/facebank/", name_person)
        if name_person not in os.listdir("face_database/facebank/"):
            os.makedirs(save_folder, exist_ok=True)

            image_path = os.path.join(save_folder, img_file_add.name)
            
            with open(image_path, "wb") as f:
                f.write(img_file_add.getbuffer())
            add_to_database(save_folder)
            st.sidebar.success(f"Image saved to {image_path}")
        else:
            st.sidebar.warning(f"{name_person} already exists")
        
    else:
        st.sidebar.warning("Please upload an image and enter a person name")

st.sidebar.markdown("----")


# Recognize

img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])


if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    image = np.array(Image.open(default_img))

st.sidebar.text("original image")
st.sidebar.image(image)

st.subheader("output image")


targets, names = update_database(False)

result, image_result = recognize_face(image, targets, names)


st.image(image_result)

# Add unknown to database

if img_file_buffer is not None:
    json_data = json.loads(result)

    for index, entry in enumerate(json_data):
        if entry["name"] == "Unknown":
            face_values = entry["face"]
            img_unknown = np.array(face_values, dtype=np.uint8)
            image_unknown = Image.fromarray(img_unknown)
            # st.image(img_unknown)
            image_bytes = image_unknown.tobytes()
            img_memory = memoryview(img_unknown)
            col1, col2, col3 = st.columns(3)
    
            with col1:
                st.image(img_unknown, caption="Unknown Face")
            
            with col2:
                name_input = st.text_input("Enter name:", key=f"name_{index}")

            with col3:
                if st.button("Add", key=f"button_{index}"):
                    if name_input:
                        
                        save_folder = os.path.join("face_database/facebank/", name_input)
                        if name_input not in os.listdir("face_database/facebank/"):
                            os.makedirs(save_folder, exist_ok=True)

                            image_path = f"{save_folder}/{name_input}.jpg"
                            image_unknown.save(image_path, "JPEG")
                            add_to_database(save_folder)
                            st.success(f"Image saved to {image_path}")
                        else:
                            st.warning(f"{name_input} already exists")
                    
                    else:
                        st.sidebar.warning("Please enter a person name")


# Show items

show_items = st.sidebar.checkbox("Show Items", key="toggle_show_items")


if show_items:
    for item in names:
        st.sidebar.write(item)




