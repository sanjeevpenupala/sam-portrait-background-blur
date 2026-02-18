# Portrait Background Blur — Model Comparison

A Streamlit app comparing 7 segmentation models for portrait background blur. Select which models to run, upload a photo, and compare mask quality and blur results side-by-side.

## Supported Models

| Model | Type | Architecture | Mask Output |
|-------|------|-------------|-------------|
| [SAM3](https://huggingface.co/1038lab/sam3) | Text-prompted segmentation | SAM3 (Ultralytics) | Binary mask |
| [BiRefNet Portrait](https://huggingface.co/ZhengPeng7/BiRefNet-portrait) | Portrait segmentation | BiRefNet | Alpha matte |
| [BiRefNet HR Matting](https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting) | High-res matting | BiRefNet | Alpha matte |
| [BiRefNet Dynamic](https://huggingface.co/ZhengPeng7/BiRefNet_dynamic) | Dynamic-res segmentation | BiRefNet | Alpha matte |
| [BiRefNet Dyn Matting](https://huggingface.co/ZhengPeng7/BiRefNet_dynamic-matting) | Dynamic-res matting | BiRefNet | Alpha matte |
| [RMBG 2.0](https://huggingface.co/briaai/RMBG-2.0) | Background removal | BiRefNet (Bria-trained) | Alpha matte |
| [BEN2](https://huggingface.co/PramaLLC/BEN2) | Background eraser | BEN2 | Alpha matte |

## Features

- **Model selection** — Sidebar checkboxes to pick which models to load and run. Models download on-demand, not at startup.
- **Mask overlays** — Green highlight with contour outlines for each model.
- **Background blur** — Inpaint-then-blur compositing with adjustable blur strength.
- **Inference timing** — Per-model inference time displayed above each result.
- **Dynamic columns** — UI adapts to the number of selected models.

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Disk space** — Model weights are downloaded automatically on first use (~1-3 GB each), cached by HuggingFace Hub in `~/.cache/huggingface/`.

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/sanjeevpenupala/sam-portrait-background-blur.git
   cd sam-portrait-background-blur
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Run the app**

   ```bash
   uv run streamlit run app.py
   ```

   Opens in your browser at `http://localhost:8501`.

4. **Select models** in the sidebar, **upload an image**, and click **"Run Segmentation and Blurring"**.

## How It Works

### Segmentation

Each model produces either a binary mask (SAM3) or a continuous alpha matte (all others). The mask is used for both the overlay visualization and background blur compositing.

- **BiRefNet variants** — Bilateral reference networks. The "dynamic" variants accept arbitrary input resolutions (trained across 256-2304px). Non-dynamic variants resize to a fixed 1024x1024. Dynamic inputs are padded to the nearest multiple of 64 to satisfy the model's patch grid requirements.
- **SAM3** — Text-prompted semantic segmentation with `["person"]`. Multiple detected masks are merged into a single binary mask.
- **RMBG 2.0** — Bria's background removal model, built on BiRefNet with proprietary training data. Uses the same inference pipeline as BiRefNet.
- **BEN2** — Background eraser network. Returns an RGBA image; the alpha channel is extracted as the matte.

### Background Blur: Inpaint-Then-Blur

A naive approach (blur the original image, then composite) smears subject pixels into the background near edges, creating visible halo artifacts. This app uses an inpaint-then-blur pipeline to avoid that:

1. **Inpaint the subject region** — `cv2.inpaint` (Telea algorithm) fills the person area with surrounding background colors, producing a continuous background with no subject in it.
2. **Multi-pass Gaussian blur** — The inpainted image is blurred 3x with the user-selected kernel size. Because the subject is already removed, there's no color bleeding at edges.
3. **Alpha composite** — `result = original * alpha + blurred * (1 - alpha)`. The sharp original subject is placed on top of the clean blurred background.

For binary masks (SAM3), the mask is Gaussian-blurred to create feathered edges before compositing. For alpha mattes (BiRefNet, RMBG, BEN2), the model's continuous values handle blending natively.

### Mask Overlay

Detected masks are thresholded to binary (>0.5), cleaned with morphological closing, and rendered as a semi-transparent green overlay with external contour outlines.

## Performance Notes

- **GPU (CUDA)** — Inference takes a few seconds per model.
- **Apple Silicon (MPS)** — PyTorch MPS backend is used automatically when available.
- **CPU only** — Works but slower (10-30+ seconds per model).

## Project Structure

```
portrait-background-blur/
├── app.py              # Streamlit app (models, segmentation, blur, UI)
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Locked dependency versions
└── .python-version     # Python version (3.13)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `transformers` | BiRefNet / RMBG model loading and inference |
| `ultralytics` | SAM3 model interface |
| `ben2` | BEN2 model interface |
| `opencv-python-headless` | Mask processing, inpainting, contour drawing, blur |
| `Pillow` | Image I/O |
| `huggingface-hub` | Auto-download model weights |
| `torch` / `torchvision` | Model backend and image transforms |
| `einops` / `kornia` | Required by BiRefNet (`trust_remote_code=True`) |

## Troubleshooting

**"Failed to run [model]"**
- Check your internet connection (models download on first use).
- Clear partial cache: `rm -rf ~/.cache/huggingface/hub/models--<org>--<model>/`

**"No humans detected"**
- BiRefNet variants use a 0.1 alpha threshold to detect presence. SAM3 uses a confidence threshold of 0.25.
- Try a different image with more clearly visible people.

**BiRefNet Dynamic error on certain image sizes**
- The dynamic variants require input dimensions compatible with their internal patch grid. The app automatically pads to the nearest multiple of 64, but extremely unusual aspect ratios may still cause issues.

**Slow inference**
- Expected on CPU. Use a CUDA GPU for faster results.
- Dynamic BiRefNet variants process at full input resolution — larger images take longer.
