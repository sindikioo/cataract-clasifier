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

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Cataract Maturity Classification",
    page_icon="👁️",
    layout="wide"
)

# ==============================
# CSS
# ==============================
st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #0f172a, #1d4ed8);
    padding: 45px;
    border-radius: 20px;
    margin-bottom: 30px;
}

.hero h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #dbeafe;
    font-size: 18px;
    line-height: 1.6;
}

/* SECTION TITLE */
.section-title {
    color: #1d4ed8;
    font-size: 25px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* INFO BOX */
.info-box {
    background-color: #eff6ff;
    border-left: 5px solid #2563eb;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 25px;
}

.info-box h3 {
    color: #1e40af;
    margin-bottom: 5px;
}

.info-box p {
    color: #475569;
    line-height: 1.6;
}

/* RESULT */
.result-box {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
}

.result-label {
    color: #166534;
    font-size: 16px;
}

.result-value {
    color: #15803d;
    font-size: 32px;
    font-weight: 800;
    margin: 10px 0;
}

.confidence-box {
    background-color: white;
    border: 1px solid #e2e8f0;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
}

.confidence-label {
    color: #1d4ed8;
    font-size: 16px;
}

.confidence-value {
    color: #1d4ed8;
    font-size: 40px;
    font-weight: 800;
    margin: 10px 0;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# HERO
# ==============================

st.markdown("""
<div class="hero">
    <h1>👁️ Cataract Maturity<br>Classification</h1>

    <p>
        Upload an eye image to predict the maturity level
        of cataracts using a Deep Learning model.
    </p>
</div>
""", unsafe_allow_html=True)


# ==============================
# ABOUT
# ==============================

st.markdown("""
<div class="info-box">

<h3>ℹ️ About This Application</h3>

<p>
This application uses a Deep Learning model to classify
cataract maturity levels based on an uploaded eye image.
</p>

</div>
""", unsafe_allow_html=True)


# ==============================
# UPLOAD
# ==============================

st.markdown(
    '<div class="section-title">① Upload Eye Image</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.write("### Select an Eye Image")

    st.caption(
        "Supported formats: JPG, JPEG, PNG"
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )


with col2:

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Eye Image",
            use_container_width=True
        )

    else:

        st.info(
            "👁️ Image preview will appear here."
        )


# ==============================
# PREDICTION
# ==============================

if uploaded_file is not None:

    st.markdown(
        '<div class="section-title">② Prediction Result</div>',
        unsafe_allow_html=True
    )

    # Resize image
    img_resized = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # Convert to array
    img_array = np.array(
        img_resized
    )

    # Add batch dimension
    img_input = np.expand_dims(
        img_array.astype(np.float32),
        axis=0
    )

    # Preprocessing
    img_input = preprocess_input(
        img_input
    )

    # Prediction
    with st.spinner("Analyzing the image..."):

        preds = model.predict(
            img_input,
            verbose=0
        )

        pred_class = np.argmax(preds)

        confidence = float(
            np.max(preds)
        )


    # ==============================
    # RESULT CARDS
    # ==============================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="result-box">

            <div class="result-label">
                Cataract Maturity Level
            </div>

            <div class="result-value">
                {class_names[pred_class]}
            </div>

            <div>
                Model confidence:
                <b>{confidence * 100:.1f}%</b>
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown(f"""
        <div class="confidence-box">

            <div class="confidence-label">
                Model Confidence
            </div>

            <div class="confidence-value">
                {confidence * 100:.1f}%
            </div>

            <div>
                Confidence of the predicted class
            </div>

        </div>
        """, unsafe_allow_html=True)


    # ==============================
    # CLASS PROBABILITIES
    # ==============================

    st.write("### Class Probabilities")

    for i, class_name in enumerate(class_names):

        probability = float(
            preds[0][i]
        )

        st.write(
            f"**{class_name}** — "
            f"{probability * 100:.1f}%"
        )

        st.progress(
            probability
        )


# ==============================
# INFORMATION
# ==============================

st.markdown("""
<div class="info-box">

<h3>💡 Image Guidelines</h3>

<p>
For better prediction performance, make sure the uploaded
eye image is clear, properly focused, and has sufficient lighting.
</p>

</div>
""", unsafe_allow_html=True)

