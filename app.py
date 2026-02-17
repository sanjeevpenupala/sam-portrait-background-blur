import streamlit as st
from huggingface_hub import hf_hub_download
from ultralytics.models.sam import SAM3SemanticPredictor


@st.cache_resource
def load_model():
    """Download SAM3 weights and initialize the predictor."""
    model_path = hf_hub_download(repo_id="1038lab/sam3", filename="sam3.pt")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
    )
    return SAM3SemanticPredictor(overrides=overrides)


st.set_page_config(page_title="Human Segmentation", layout="wide")
st.title("Human Segmentation with SAM3")

with st.spinner("Loading SAM3 model..."):
    predictor = load_model()

st.success("Model loaded.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Segmentation")
        st.info("Segmentation will appear here.")
