import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

IMG_SIZE = 224

# PENTING: urutan di bawah ini HARUS sama persis dengan urutan yang dipakai
# model saat training. Biasanya ini urutan alfabetis dari nama folder dataset.
# Cek ulang di notebook Colab kamu dengan: train_generator.class_indices
class_names = ["Immature", "Mature", "Normal"]


@st.cache_resource  # biar model cuma di-load sekali, bukan setiap ganti gambar
def get_model():
    return load_model("bestmodel.h5")


model = get_model()

st.title("Klasifikasi Tingkat Maturitas Katarak")
st.write("Cataract Maturity Classification — Upload gambar mata untuk memprediksi tingkat maturitas katarak")

uploaded_file = st.file_uploader("Upload gambar mata", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_input = preprocess_input(img_input)

    with st.spinner("Memprediksi..."):
        preds = model.predict(img_input, verbose=0)
        pred_class = np.argmax(preds)
        confidence = float(np.max(preds))

    st.subheader("Hasil Prediksi")
    st.write(f"**{class_names[pred_class]}** ({confidence*100:.1f}% yakin)")
