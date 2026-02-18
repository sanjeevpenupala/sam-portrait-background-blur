import tempfile
import time
from unittest.mock import patch

import cv2
import numpy as np
import streamlit as st
import torch
from ben2 import BEN_Base
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageSegmentation
from ultralytics.models.sam import SAM3SemanticPredictor


def get_available_devices():
    """Return list of available compute devices."""
    devices = ["Auto"]
    if torch.cuda.is_available():
        devices.append("CUDA")
    if torch.backends.mps.is_available():
        devices.append("MPS")
    devices.append("CPU")
    return devices


def resolve_device(choice: str) -> torch.device:
    """Resolve a device choice string to a torch.device."""
    if choice == "Auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice.lower())


def merge_masks(masks: np.ndarray, h: int, w: int) -> np.ndarray:
    """Merge multiple masks into a single binary mask at (h, w) resolution."""
    merged = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
        merged = np.maximum(merged, mask.astype(np.uint8))
    return merged


def render_segmentation_overlay(original: np.ndarray, merged: np.ndarray) -> np.ndarray:
    """Draw green overlay and contours on the image from a merged binary mask."""
    overlay = original.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)

    bool_mask = closed.astype(bool)
    color = (0, 255, 0, 255) if overlay.ndim == 3 and overlay.shape[2] == 4 else (0, 255, 0)
    overlay[bool_mask] = (overlay[bool_mask] * 0.5 + np.array(color) * 0.5).astype(
        np.uint8
    )

    dilated = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def blur_background(
    original: np.ndarray, merged: np.ndarray, blur_strength: int = 51, soft_mask: bool = False
) -> np.ndarray:
    """Blur the background using inpaint-then-blur to avoid halo artifacts.

    1. Inpaint the subject region so the background is continuous
    2. Blur the inpainted image (no subject pixels to smear)
    3. Composite the sharp subject on top using the feathered mask
    """
    h, w = original.shape[:2]

    if merged.shape != (h, w):
        merged = cv2.resize(merged, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to float alpha mask [0, 1]
    alpha = merged.astype(np.float32)
    if alpha.max() > 1:
        alpha = alpha / 255.0

    # Feather the edges with Gaussian blur on the mask itself (skip for soft alpha mattes)
    if not soft_mask:
        alpha = cv2.GaussianBlur(alpha, (15, 15), sigmaX=5)

    # Create binary inpaint mask: dilate to cover edge fringe
    inpaint_mask = (alpha > 0.1).astype(np.uint8) * 255
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    inpaint_mask = cv2.dilate(inpaint_mask, dilate_kernel, iterations=1)

    # Inpaint the subject region with surrounding background colors
    inpainted = cv2.inpaint(original, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Blur the inpainted image (subject is gone, so no color bleeding)
    blurred = inpainted
    for _ in range(3):
        blurred = cv2.GaussianBlur(blurred, (blur_strength, blur_strength), sigmaX=0)

    # Composite: sharp subject on top of clean blurred background
    alpha_3ch = alpha[:, :, np.newaxis]
    result = (original * alpha_3ch + blurred * (1 - alpha_3ch)).astype(np.uint8)

    return result


def cut_paste_blur(original: np.ndarray, mask: np.ndarray, blur_strength: int = 51) -> np.ndarray:
    """Cut person with raw mask, inpaint+blur background, paste person back."""
    h, w = original.shape[:2]

    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    binary = (mask > 0.5).astype(np.uint8)

    # Inpaint the subject region, then blur the clean background
    inpaint_mask = cv2.dilate(binary * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    inpainted = cv2.inpaint(original, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    blurred = inpainted
    for _ in range(3):
        blurred = cv2.GaussianBlur(blurred, (blur_strength, blur_strength), sigmaX=0)

    # Paste sharp subject on top of blurred background
    binary_3ch = binary[:, :, np.newaxis]
    result = (original * binary_3ch + blurred * (1 - binary_3ch)).astype(np.uint8)

    return result


@st.cache_resource
def load_birefnet(device_name: str, repo_id: str = "ZhengPeng7/BiRefNet-portrait", use_float: bool = False):
    """Download and initialize a BiRefNet-family model."""
    dev = torch.device(device_name)
    model = AutoModelForImageSegmentation.from_pretrained(
        repo_id, trust_remote_code=True
    )
    if use_float:
        model.float()
    model.to(dev).eval()
    return model


@st.cache_resource
def load_rmbg2(device_name: str):
    """Download and initialize Bria RMBG 2.0 model.

    Monkey-patches torch.linspace and adds missing post_init to work around
    RMBG 2.0's incompatibility with transformers 5.x meta device initialization.
    See: https://github.com/ZhengPeng7/BiRefNet/issues/285
    """
    _orig_linspace = torch.linspace

    def _cpu_linspace(*args, **kwargs):
        kwargs.pop("device", None)
        return _orig_linspace(*args, **kwargs, device="cpu")

    with patch("torch.linspace", _cpu_linspace):
        config = AutoConfig.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
        model_class = type(
            AutoModelForImageSegmentation.from_config(config, trust_remote_code=True)
        )

        orig_init = model_class.__init__

        def _patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()

        model_class.__init__ = _patched_init
        model = model_class.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)

    model.to(torch.device(device_name)).eval()
    return model


@st.cache_resource
def load_sam3(device_name: str):
    """Download SAM3 weights and initialize the predictor."""
    model_path = hf_hub_download(repo_id="1038lab/sam3", filename="sam3.pt")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        device=device_name,
    )
    return SAM3SemanticPredictor(overrides=overrides)


@st.cache_resource
def load_ben2(device_name: str):
    """Download and initialize BEN2 model."""
    dev = torch.device(device_name)
    model = BEN_Base.from_pretrained("PramaLLC/BEN2")
    model.to(dev).eval()
    return model


birefnet_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ResizeToMultiple:
    """Resize so both dimensions are divisible by `multiple`, preserving aspect ratio."""

    def __init__(self, multiple: int = 64):
        self.multiple = multiple

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = (w + self.multiple - 1) // self.multiple * self.multiple
        new_h = (h + self.multiple - 1) // self.multiple * self.multiple
        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h), Image.BILINEAR)
        return img


birefnet_dynamic_transform = transforms.Compose([
    ResizeToMultiple(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run_birefnet_segmentation(model, image: Image.Image, device: torch.device, transform=None):
    """Run BiRefNet segmentation. Returns (overlay, alpha_matte, elapsed) or (None, None, elapsed)."""
    if transform is None:
        transform = birefnet_transform
    original = np.array(image)
    h, w = original.shape[:2]

    input_tensor = transform(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid()
    elapsed = time.perf_counter() - start

    # Squeeze to 2D and resize back to original dimensions
    alpha_matte = preds.squeeze().cpu().numpy()
    alpha_matte = cv2.resize(alpha_matte, (w, h), interpolation=cv2.INTER_LINEAR)

    # Check if any person detected (meaningful alpha values)
    if alpha_matte.max() < 0.1:
        return None, None, elapsed

    # Binary mask for overlay visualization
    binary_mask = (alpha_matte > 0.5).astype(np.uint8)
    overlay = render_segmentation_overlay(original, binary_mask)

    return Image.fromarray(overlay), alpha_matte, elapsed


def run_sam3_segmentation(predictor, image: Image.Image):
    """Run SAM3 segmentation. Returns (overlay, merged_mask, elapsed) or (None, None, elapsed)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.convert("RGB").save(f, format="JPEG")
        temp_path = f.name

    original = np.array(image)
    h, w = original.shape[:2]

    start = time.perf_counter()
    predictor.set_image(temp_path)
    results = predictor(text=["person"])
    elapsed = time.perf_counter() - start

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        merged = merge_masks(masks, h, w)
        overlay = render_segmentation_overlay(original, merged)
        return Image.fromarray(overlay), merged, elapsed
    return None, None, elapsed


def run_ben2_segmentation(model, image: Image.Image):
    """Run BEN2 segmentation. Returns (overlay, alpha_matte, elapsed) or (None, None, elapsed)."""
    image = image.copy()
    original = np.array(image)
    h, w = original.shape[:2]

    start = time.perf_counter()
    with torch.no_grad():
        result = model.inference(image)
    elapsed = time.perf_counter() - start

    # Extract alpha channel from RGBA output (0-255) and normalize to 0-1
    alpha_matte = np.array(result.split()[3]).astype(np.float32) / 255.0

    if alpha_matte.max() < 0.1:
        return None, None, elapsed

    binary_mask = (alpha_matte > 0.5).astype(np.uint8)
    overlay = render_segmentation_overlay(original, binary_mask)

    return Image.fromarray(overlay), alpha_matte, elapsed


TRANSFORM_MAP = {
    "birefnet": birefnet_transform,
    "birefnet_dynamic": birefnet_dynamic_transform,
}


def run_inference(spec, model, image, device):
    """Dispatch inference to the correct runner based on spec."""
    runner = spec["runner"]
    if runner == "birefnet":
        transform = TRANSFORM_MAP.get(spec.get("transform", "birefnet"))
        return run_birefnet_segmentation(model, image, device, transform=transform)
    if runner == "sam3":
        return run_sam3_segmentation(model, image)
    if runner == "ben2":
        return run_ben2_segmentation(model, image)
    raise ValueError(f"Unknown runner: {runner}")


MODEL_REGISTRY = [
    {
        "key": "sam3",
        "label": "SAM3",
        "loader": load_sam3,
        "loader_kwargs": {},
        "runner": "sam3",
        "transform": None,
        "soft_mask": False,
    },
    {
        "key": "birefnet_portrait",
        "label": "BiRefNet Portrait",
        "loader": load_birefnet,
        "loader_kwargs": {"repo_id": "ZhengPeng7/BiRefNet-portrait"},
        "runner": "birefnet",
        "transform": "birefnet",
        "soft_mask": True,
    },
    {
        "key": "birefnet_hr_matting",
        "label": "BiRefNet HR Matting",
        "loader": load_birefnet,
        "loader_kwargs": {"repo_id": "ZhengPeng7/BiRefNet_HR-matting", "use_float": True},
        "runner": "birefnet",
        "transform": "birefnet",
        "soft_mask": True,
    },
    {
        "key": "birefnet_dynamic",
        "label": "BiRefNet Dynamic",
        "loader": load_birefnet,
        "loader_kwargs": {"repo_id": "ZhengPeng7/BiRefNet_dynamic", "use_float": True},
        "runner": "birefnet",
        "transform": "birefnet_dynamic",
        "soft_mask": False,
    },
    {
        "key": "birefnet_dynamic_matting",
        "label": "BiRefNet Dyn Matting",
        "loader": load_birefnet,
        "loader_kwargs": {"repo_id": "ZhengPeng7/BiRefNet_dynamic-matting", "use_float": True},
        "runner": "birefnet",
        "transform": "birefnet_dynamic",
        "soft_mask": True,
    },
    {
        "key": "rmbg2",
        "label": "RMBG 2.0",
        "loader": load_rmbg2,
        "loader_kwargs": {},
        "runner": "birefnet",
        "transform": "birefnet",
        "soft_mask": True,
    },
    {
        "key": "ben2",
        "label": "BEN2",
        "loader": load_ben2,
        "loader_kwargs": {},
        "runner": "ben2",
        "transform": None,
        "soft_mask": False,
    },
]


st.set_page_config(page_title="Portrait Background Blur", layout="wide")
st.title("Portrait Background Blur — Model Comparison")

available_devices = get_available_devices()
device_choice = st.sidebar.selectbox("Compute Device", available_devices)
device = resolve_device(device_choice)
device_name = str(device)
st.sidebar.caption(f"Active: **{device_name}**")

st.sidebar.markdown("**Models to compare**")
active_specs = [
    spec for spec in MODEL_REGISTRY
    if st.sidebar.checkbox(spec["label"], value=True, key=f"model_{spec['key']}")
]

if not active_specs:
    st.warning("Select at least one model from the sidebar.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)

    st.image(image, caption="Original", width=400)

    if not st.button("Run Segmentation and Blurring", type="primary"):
        st.stop()

    models = {}
    results = {}
    for spec in active_specs:
        try:
            with st.spinner(f"Loading {spec['label']}..."):
                models[spec["key"]] = spec["loader"](device_name, **spec["loader_kwargs"])
            with st.spinner(f"Running {spec['label']}..."):
                overlay, mask, elapsed = run_inference(spec, models[spec["key"]], image, device)
                results[spec["key"]] = {"overlay": overlay, "mask": mask, "time": elapsed}
        except Exception as e:
            st.error(f"Failed to run {spec['label']}: {e}")
            results[spec["key"]] = {"overlay": None, "mask": None, "time": 0.0}

    st.divider()

    cols = st.columns(len(active_specs))
    for col, spec in zip(cols, active_specs):
        r = results[spec["key"]]
        with col:
            st.subheader(spec["label"])
            st.caption(f"Inference: {r['time']:.2f}s")
            if r["overlay"] is not None:
                st.image(r["overlay"], caption="Mask Overlay", width="stretch")
            else:
                st.warning("No humans detected.")

    any_mask = any(results[s["key"]]["mask"] is not None for s in active_specs)
    if any_mask:
        st.divider()
        blur_strength = st.slider(
            "Blur Strength", min_value=5, max_value=255, value=101, step=2
        )
        st.write("")

        blur_cols = st.columns(len(active_specs))
        for col, spec in zip(blur_cols, active_specs):
            r = results[spec["key"]]
            with col:
                if r["mask"] is not None:
                    blurred = blur_background(
                        original, r["mask"], blur_strength, soft_mask=spec["soft_mask"]
                    )
                    st.image(blurred, caption=f"{spec['label']} — Background Blur", width="stretch")

        # st.divider()
        # st.subheader("Cut / Paste Comparison")
        # st.caption(
        #     "**Above:** Alpha blending — mask values smoothly blend sharp foreground with blurred background. "
        #     "For SAM3 (binary mask), we Gaussian-blur the mask first to fake soft edges. "
        #     "For matting models (alpha matte), the model's continuous [0–1] values handle blending natively."
        # )
        # st.caption(
        #     "**Below:** Naive cut/paste — threshold all masks to hard binary (person=1, background=0), "
        #     "blur the full image, stamp the person on top. No edge blending at all. "
        #     "Identical to alpha blend for perfect masks, but exposes harsh edges on binary masks like SAM3."
        # )
        #
        # cp_cols = st.columns(len(active_specs))
        # for col, spec in zip(cp_cols, active_specs):
        #     r = results[spec["key"]]
        #     with col:
        #         if r["mask"] is not None:
        #             cp = cut_paste_blur(original, r["mask"], blur_strength)
        #             st.image(cp, caption=f"{spec['label']} — Cut/Paste", width="stretch")
