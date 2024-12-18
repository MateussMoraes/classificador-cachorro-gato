import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

st.title("Cachorro ou Gato 🐶🐱")
st.text("Envie uma imagem para saber se é um cachorro ou um gato.")

# Link para o Google Colab
link = "https://colab.research.google.com/drive/12BNnVJquI6B_PZHd2ukdRtUhyhehzZv2?usp=sharing"

st.text("O código utilizado está disponível no google colab, clique no botão abaixo.")
# Botão
if st.button("Abrir Google Colab"):
    webbrowser.open_new_tab(link)

@st.cache_resource
def load_trained_model():
    return load_model('modelo_CatsAndDogs.h5')

model = load_trained_model()

def process_image(image):
    img = image.resize((128, 128)) 
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image):
    processed_image = process_image(image)
    st.text(f"Formato da imagem processada: {processed_image.shape}") 
    prediction = model.predict(processed_image)
    st.text(f"Predição bruta: {prediction}")
    return "ÉÉ Cachorro 🐶" if prediction[0] >= 0.5 else "ÉÉ Gato 🐱"

uploaded_image = st.file_uploader("Envie uma imagem (JPEG ou PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagem carregada", use_container_width=True)

    try:
        with st.spinner("Classificando..."):
            result = predict_image(image)
        st.success(f"Resultado: {result}")
    except Exception as e:
        st.error("Erro ao processar a imagem.")
        st.text(str(e))