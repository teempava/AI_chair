import streamlit as st
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.inception_v3 import preprocess_input

model = load_model('AI_chair(r200).h5', compile=False)

target_img_shape = (128, 128)

st.subheader("การจำแนกสภาพเก้าอี้")
img_file = st.file_uploader("เปิดไฟล์ภาพ")

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # ------------------------------------------------
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(img, target_img_shape)

    test_image = img_to_array(test_image)
    test_image = preprocess_input(test_image)

    test_image = np.expand_dims(test_image, axis=0)  # (1, 128, 128, 3)

    result = model.predict(test_image)
    st.write("Model Predictions:", result)

    class_answer = np.argmax(result)
    if class_answer == 0:
        predict = 'broken'
    elif class_answer == 1:
        predict = 'good'

    st.write("predict =", predict)
    # ------------------------------------------------
    st.image(img, caption=predict, channels="RGB")
