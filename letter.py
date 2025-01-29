import io
import torch
import requests
import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor,TrOCRProcessor

#кэшируем модели для разспознавания
@st.cache_data
def load_model():
    return VisionEncoderDecoderModel.from_pretrained("kazars24/trocr-base-handwritten-ru")
@st.cache_data
def load_processor():
    return TrOCRProcessor.from_pretrained("kazars24/trocr-base-handwritten-ru")

# Функция распознавания объектов на изображении
def predict_step(image):
  image = image.convert("RGB")
  pixel_values = processor(images=image, return_tensors="pt").pixel_values
  generated_ids = model.generate(pixel_values)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return generated_text

# Функция обращения к API переводчика
#def translate(payload, API_URL):
	#response = requests.post(API_URL, headers=headers, json=payload )
	#return response.json

# Функция загрузки ихображения через Streamlit
def load_image():
    uploaded_file = st.file_uploader(label='Загрузите изображение:')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

# Фукнция вызова отображения переводов
#def print_predictions(preds):
    for cl in preds:
        #st.write(str(cl).replace('_'," "))
        en_text=str(cl).replace('_'," ")
        trans_ta = translate({"inputs":  [">>rus<< "+en_text,  ">>deu<< "+en_text, ],
                             "parameters":{ "src_lang":"en", "tgt_lang":"ru_RU"}
                             }, API_URL_ta)
        sleep_duration = 5
        tr_test=tuple(trans_ta())
        st.write('рус.: ', str(tr_test[0]["translation_text"]))
        st.write('нем.: ', str(tr_test[1]["translation_text"]))
        #for tt in tr_test:
        #    st.write(str(tt['translation_text']))
	    

st.title('Распознавание письменного текста с картинки')
model = load_model()	
processor = load_processor()    
            
im=load_image()
sleep_duration = 0.5
result = st.button('Распознать:')
if result:
   preds = predict_step(im)
   st.write('**На картинке:**')
   st.write(str(preds))
   #print_predictions(preds)