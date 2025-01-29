import io
import torch
import requests
import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel,TrOCRProcessor
from transformers import pipeline



#кэшируем модель
@st.cache_data
#def load_model():
    #return VisionEncoderDecoderModel.from_pretrained("kazars24/trocr-base-handwritten-ru")
#def load_processor():
    #return TrOCRProcessor.from_pretrained("kazars24/trocr-base-handwritten-ru")

# Функция распознавания объектов на изображении
#def predict_step(image):
  #image = image.convert("RGB")
  #pixel_values = processor(images=image, return_tensors="pt").pixel_values
  #generated_ids = model.generate(pixel_values)
  #generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  #return generated_text

# Функция загрузки ихображения через Streamlit
def load_image():
    uploaded_file = st.file_uploader(label='Загрузите изображение:')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

st.title('Распознавание письменного текста с картинки')
#model = load_model()	
#processor = load_processor()    
image_to_text = pipeline("image-to-text", model="kazars24/trocr-base-handwritten-ru")

im=load_image()
sleep_duration = 0.5
result = st.button('Распознать:')
if result:
   preds = image_to_text(im)
   st.write('**На картинке:**')
   st.write(str(preds))