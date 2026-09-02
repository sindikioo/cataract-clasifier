import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

IMG_SIZE = 224

class_names = ["Immature", "Mature", "Normal"]


@st.cache_resource  
def get_model():
    return load_model("bestmodel.h5")


model = get_model()

st.title("Cataract Maturity Classification")
st.write("Upload an eye image to claasify the maturity level of cataracts using a Deep Learning model")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image uploaded", use_container_width=True)

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    img_input = np.expand_dims(img_array.astype(np.float32), axis=0)
    img_input = preprocess_input(img_input)

    with st.spinner("Predicting....."):
        preds = model.predict(img_input, verbose=0)
        pred_class = np.argmax(preds)
        confidence = float(np.max(preds))

    st.subheader("Result")
    st.write(f"**{class_names[pred_class]}** ({confidence*100:.1f}% confidence)")
