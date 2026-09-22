from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.predict import MODEL_PATH, predict_image


st.set_page_config(
    page_title="EMNIST AI",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 8% 5%, rgba(99,91,255,.18), transparent 28rem),
                    radial-gradient(circle at 92% 10%, rgba(236,72,153,.12), transparent 25rem),
                    linear-gradient(135deg, #eef1f8, #f8f9fd 50%, #eef4f8);
        color: #171a2b;
        font-family: 'DM Sans', sans-serif;
    }

    .block-container { max-width: 820px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #171a2b !important; font-family: 'Space Grotesk', sans-serif !important; }
    h1 { font-size: clamp(2.4rem, 7vw, 4.5rem) !important; line-height: 1 !important; }
    [data-testid='stMarkdownContainer'] p, [data-testid='stCaptionContainer'] p { color: #687085; }
    [data-testid='stVerticalBlockBorderWrapper'] {
        background: rgba(255,255,255,.48);
        border: 1px solid rgba(255,255,255,.8);
        border-radius: 26px;
        box-shadow: 0 18px 45px rgba(35,40,75,.10), inset 0 1px 1px rgba(255,255,255,.85);
        backdrop-filter: blur(18px);
    }
    [data-testid='stCanvas'] {
        border: 2px solid rgba(99,91,255,.28) !important;
        border-radius: 20px !important;
        box-shadow: 0 15px 35px rgba(35,40,75,.10);
        overflow: hidden;
    }
    .stButton > button {
        min-height: 3rem;
        border: 1px solid rgba(99,91,255,.25);
        border-radius: 14px;
        background: linear-gradient(135deg, #635bff, #8b5cf6);
        color: white;
        font-weight: 700;
        box-shadow: 0 10px 22px rgba(99,91,255,.20);
    }
    [data-testid='stMetric'] {
        background: rgba(255,255,255,.76);
        border: 1px solid rgba(255,255,255,.86);
        border-radius: 20px;
        box-shadow: 0 12px 32px rgba(35,40,75,.08);
    }
    [data-testid='stMetricValue'] { color: #635bff; font-family: 'Space Grotesk', sans-serif; }
    [data-testid='stProgressBar'] > div > div { background: linear-gradient(90deg, #635bff, #8b5cf6, #ec4899); }
    [data-testid='stImage'] img { border-radius: 18px; box-shadow: 0 12px 30px rgba(35,40,75,.08); }
    [data-testid='stAlert'] { border-radius: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "predicted" not in st.session_state:
    st.session_state.predicted = False


def store_prediction(image: Image.Image) -> None:
    label, confidence, normalized = predict_image(image)
    st.session_state.predicted = True
    st.session_state.prediction = label
    st.session_state.confidence = confidence
    st.session_state.normalized = normalized
    st.session_state.input_image = image


st.title("✍️ EMNIST AI")
st.subheader("Your handwriting. Understood by AI.")
st.write("Draw one character below, then ask the CNN to recognize it.")

feature_col1, feature_col2, feature_col3 = st.columns(3)
feature_col1.info("🧠 CNN model")
feature_col2.success("📚 EMNIST Balanced")
feature_col3.warning("⚡ Live prediction")
st.divider()

if not Path(MODEL_PATH).exists():
    st.error("Model file not found. Run the training notebook first.")
    st.stop()

st.header("01  Choose your input")
st.caption("Upload one character image or draw one clearly in the canvas.")

draw_tab, upload_tab = st.tabs(["✏️ Draw character", "📁 Upload image"])

with upload_tab:
    with st.container(border=True):
        st.subheader("Upload handwritten character")
        st.caption("Supported formats: PNG, BMP, and WebP.")
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "bmp", "webp"],
            key="character_upload",
        )

        if uploaded is not None:
            uploaded_image = Image.open(uploaded).convert("L")
            st.image(uploaded_image, caption="Uploaded character", width=280)
            if st.button("🔮 Predict uploaded image", key="upload_predict", use_container_width=True):
                try:
                    store_prediction(uploaded_image)
                    st.success("Prediction complete")
                except Exception as error:
                    st.error("Prediction failed. Check the model and dependencies.")
                    st.exception(error)

with draw_tab:
    with st.container(border=True):
        st.subheader("Draw your character")
        st.caption("Use your mouse or touchscreen. Draw one centered character.")
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=16,
            stroke_color="#635bff",
            background_color="#ffffff",
            width=420,
            height=420,
            drawing_mode="freedraw",
            display_toolbar=True,
            update_streamlit=True,
            key="character_canvas",
        )

        canvas_image = None
        if canvas.image_data is not None:
            canvas_rgb = canvas.image_data[:, :, :3].astype("uint8")
            ink_pixels = np.any(canvas_rgb < 245, axis=2)
            if np.count_nonzero(ink_pixels) > 25:
                canvas_image = Image.fromarray(canvas_rgb).convert("L")
                st.image(canvas_image, caption="Drawing preview", width=280)

        if canvas_image is None:
            st.info("Draw on the canvas to enable prediction.")
        elif st.button("🔮 Predict drawn character", key="draw_predict", use_container_width=True):
            try:
                store_prediction(canvas_image)
                st.success("Prediction complete")
            except Exception as error:
                st.error("Prediction failed. Check the model and dependencies.")
                st.exception(error)

if st.session_state.predicted:
    st.divider()
    st.header("02  Prediction")

    result_col, confidence_col = st.columns(2)
    with result_col:
        st.metric("Predicted character", st.session_state.prediction)
    with confidence_col:
        st.metric("Confidence", f"{st.session_state.confidence:.2%}")

    confidence = st.session_state.confidence
    st.progress(float(confidence), text="Prediction confidence")
    if confidence >= 0.90:
        st.success("Very high confidence")
    elif confidence >= 0.70:
        st.info("Good confidence")
    else:
        st.warning("Low confidence. Draw the character larger and more clearly.")

    st.subheader("What the model sees")
    image_col, normalized_col = st.columns(2)
    with image_col:
        st.image(st.session_state.input_image, caption="Original drawing", width=260)
    with normalized_col:
        st.image(st.session_state.normalized, caption="Normalized 28 × 28 input", width=220)

st.divider()
st.caption("EMNIST Balanced • Convolutional Neural Network • Streamlit")
