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

import streamlit as st
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cataract Maturity Classification",
    page_icon="👁️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

    /* ---------- General ---------- */

    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }


    /* ---------- Hero Section ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e3a8a
        );

        padding: 45px 50px;
        border-radius: 20px;
        margin-bottom: 30px;

        box-shadow:
            0 10px 30px rgba(15, 23, 42, 0.15);
    }

    .hero-icon {
        font-size: 55px;
        margin-bottom: 10px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: white;
        margin-bottom: 10px;
    }

    .main-subtitle {
        font-size: 18px;
        color: #dbeafe;
        line-height: 1.6;
    }


    /* ---------- Info Box ---------- */

    .info-box {
        background-color: #eff6ff;

        border-left: 5px solid #2563eb;

        padding: 20px 25px;

        border-radius: 12px;

        margin-bottom: 25px;
    }

    .info-title {
        font-size: 20px;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 5px;
    }

    .info-text {
        color: #475569;
        font-size: 16px;
        line-height: 1.6;
    }


    /* ---------- Section Title ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #1d4ed8;
        margin-top: 20px;
        margin-bottom: 15px;
    }


    /* ---------- Cards ---------- */

    .card {
        background-color: white;

        padding: 28px;

        border-radius: 18px;

        border: 1px solid #e2e8f0;

        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.06);

        margin-bottom: 25px;
    }


    /* ---------- Prediction ---------- */

    .prediction-box {
        background-color: #f0fdf4;

        border: 1px solid #bbf7d0;

        padding: 30px;

        border-radius: 15px;

        text-align: center;
    }

    .prediction-label {
        color: #166534;

        font-size: 16px;

        margin-bottom: 8px;
    }

    .prediction-result {
        color: #15803d;

        font-size: 32px;

        font-weight: 800;
    }

    .confidence {
        color: #475569;

        font-size: 17px;

        margin-top: 10px;
    }


    /* ---------- Confidence Card ---------- */

    .confidence-box {
        background-color: white;

        border: 1px solid #e2e8f0;

        padding: 30px;

        border-radius: 15px;

        text-align: center;
    }

    .confidence-title {
        color: #1d4ed8;

        font-size: 16px;

        font-weight: 600;
    }

    .confidence-value {
        color: #1d4ed8;

        font-size: 40px;

        font-weight: 800;

        margin: 10px 0;
    }


    /* ---------- File Uploader ---------- */

    [data-testid="stFileUploader"] {
        background-color: #f8fafc;

        border: 2px dashed #93c5fd;

        border-radius: 15px;

        padding: 15px;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;

        color: #64748b;

        font-size: 14px;

        padding: 25px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">
        👁️
    </div>

    <div class="main-title">
        Cataract Maturity<br>
        Classification
    </div>

    <div class="main-subtitle">
        Upload an eye image to predict the maturity level
        of cataracts using a Deep Learning model.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# ABOUT THE APPLICATION
# =========================================================

st.markdown("""
<div class="info-box">

    <div class="info-title">
        ℹ️ About This Application
    </div>

    <div class="info-text">
        This application uses a Deep Learning model to classify
        cataract maturity levels based on an uploaded eye image.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown("""
<div class="section-title">
    ① Upload Eye Image
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(
    [1, 1],
    gap="large"
)


# ---------- Upload ----------

with col1:

    st.write(
        "**Select an eye image to perform the classification.**"
    )

    st.caption(
        "Supported formats: JPG, JPEG, PNG"
    )

    uploaded_file = st.file_uploader(
        "Upload eye image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )


# ---------- Preview ----------

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

        st.markdown("""
        <div style="
            height:300px;
            display:flex;
            align-items:center;
            justify-content:center;

            background:#f8fafc;

            border:2px dashed #cbd5e1;

            border-radius:15px;

            color:#64748b;

            font-size:16px;
        ">

            👁️

            <span style="
                margin-left:8px;
            ">
                Image preview will appear here
            </span>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="section-title">
        ② Prediction Result
    </div>
    """, unsafe_allow_html=True)


    # =====================================================
    # PREPROCESSING
    # =====================================================

    img_resized = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    img_array = np.array(
        img_resized
    )

    img_input = np.expand_dims(
        img_array.astype(np.float32),
        axis=0
    )

    img_input = preprocess_input(
        img_input
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    with st.spinner(
        "Analyzing the image..."
    ):

        preds = model.predict(
            img_input,
            verbose=0
        )

        pred_class = np.argmax(
            preds
        )

        confidence = float(
            np.max(preds)
        )


    # =====================================================
    # RESULT CARDS
    # =====================================================

    col1, col2 = st.columns(
        2,
        gap="large"
    )


    # ---------- Prediction ----------

    with col1:

        st.markdown(f"""
        <div class="prediction-box">

            <div class="prediction-label">
                Cataract Maturity Level
            </div>

            <div class="prediction-result">
                {class_names[pred_class]}
            </div>

            <div class="confidence">
                ✓ Model confidence:
                <b>{confidence*100:.1f}%</b>
            </div>

        </div>
        """, unsafe_allow_html=True)


    # ---------- Confidence ----------

    with col2:

        st.markdown(f"""
        <div class="confidence-box">

            <div class="confidence-title">
                Model Confidence
            </div>

            <div class="confidence-value">
                {confidence*100:.1f}%
            </div>

            <div style="
                color:#64748b;
                font-size:14px;
            ">
                Confidence of the predicted class
            </div>

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        "### Class Probabilities"
    )


    for i, class_name in enumerate(
        class_names
    ):

        probability = float(
            preds[0][i]
        )

        st.write(
            f"**{class_name}** — "
            f"{probability*100:.1f}%"
        )

        st.progress(
            probability
        )


# =========================================================
# INFORMATION
# =========================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

    <div class="info-title">
        💡 Image Guidelines
    </div>

    <div class="info-text">
        For better prediction performance, make sure the
        uploaded eye image is clear, properly focused, and
        has sufficient lighting.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    © 2026 Cataract Maturity Classification
    &nbsp;•&nbsp;
    Built with Streamlit

</div>
""", unsafe_allow_html=True)
