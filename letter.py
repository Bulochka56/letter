import io
import torch
import requests
import streamlit as st
from PIL import Image

from transformers import pipeline

#кэшируем модель
@st.cache_data
def load_model():
    return pipeline("image-to-text", model="kazars24/trocr-base-handwritten-ru")

# Функция загрузки ихображения через Streamlit
def load_image():
    uploaded_file = st.file_uploader(label='Загрузите изображение:')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

st.title('Распознавание письменного текста с изображения')    
image_to_text = load_model()

im=load_image()
sleep_duration = 0.5
result = st.button('Распознать:')
if result:
   preds = image_to_text(im)
   st.write('**На картинке:**')
   st.write(str(preds[0]["generated_text"]))