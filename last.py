"""
CT Bowel Injury Detection Demo - Streamlit Web Application
===========================================================
A prototype AI system for assessing bowel injury risk from abdominal CT scans.
Uses a 2.5D CNN-GRU model (ResNet18 + GRU) trained on RSNA 2023 data.

WARNING: Educational prototype only. NOT for clinical diagnosis.
"""

import io
import os
import sys
import tempfile
import zipfile
import shutil
import datetime
import numpy as np
import streamlit as st
import torch
import gdown
import pydicom
from scipy.ndimage import zoom
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# =========================
# Configuration
# =========================
MODEL_PATH = "best_bowel_injury_model.pth"
GDRIVE_FILE_ID = "1-awchgMTBa9Ra7jYzlKzccN8MKeUvOs_"

DEFAULT_NUM_STEPS = 32
DEFAULT_NUM_SLICES_PER_STEP = 3
TARGET_SHAPE = (96, 256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Safety limits
MAX_NPY_FILE_SIZE_MB = 500
MAX_ZIP_FILE_SIZE_MB = 500
MAX_ZIP_EXTRACTED_SIZE_MB = 2000
MAX_DICOM_SLICES = 1000


# =========================
# Model definition
# =========================
class CNNGRUClassifier(torch.nn.Module):
    """2.5D classifier: ResNet18 (spatial features) + GRU (slice sequence)."""

    def __init__(self, cnn_name: str = "resnet18", hidden_size: int = 256, num_classes: int = 1) -> None:
        super().__init__()
        if cnn_name == "resnet18":
            # Use pretrained weights to match training script's state_dict keys
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.cnn = torch.nn.Sequential(*(list(model.children())[:-1]))
            cnn_out_channels = 512
        else:
            raise ValueError(f"Unsupported cnn_name: {cnn_name}")
        self.gru = torch.nn.GRU(
            input_size=cnn_out_channels, hidden_size=hidden_size,
            num_layers=1, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, c, h, w = x.shape
        x = x.view(batch_size * time_steps, c, h, w)
        features = self.cnn(x)
        features = features.view(batch_size, time_steps, -1)
        output, _ = self.gru(features)
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits.squeeze(dim=-1)


# =========================
# Model download
# =========================
def ensure_model_downloaded():
    """Download model weights from Google Drive if not present locally."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    tmp_path = os.path.join(tempfile.gettempdir(), MODEL_PATH)

    with st.spinner("Downloading model from Google Drive..."):
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        try:
            gdown.download(url, tmp_path, quiet=False, fuzzy=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model: {e}\n"
                "Please check your internet connection or manually place "
                f"'{MODEL_PATH}' in the app directory."
            )

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 100_000:
        raise RuntimeError(
            "Download failed or file is too small (possibly HTML error page). "
            "Check that the Google Drive file is shared as 'Anyone with the link'."
        )

    with open(tmp_path, "rb") as f:
        head = f.read(200).lower()
    if b"<html" in head or b"google drive" in head:
        os.remove(tmp_path)
        raise RuntimeError(
            "Downloaded HTML instead of model weights. "
            "The Google Drive link may have expired or require permission."
        )

    shutil.move(tmp_path, MODEL_PATH)


@st.cache_resource
def load_model():
    """Load and cache the CNN-GRU model."""
    ensure_model_downloaded()
    model = CNNGRUClassifier(cnn_name="resnet18", hidden_size=256).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# =========================
# Input validation
# =========================
def validate_npy_volume(vol: np.ndarray) -> tuple[bool, str]:
    """Validate that an uploaded .npy array is a valid CT volume."""
    if vol.ndim != 3:
        return False, f"Expected 3D array (Z, H, W), got {vol.ndim}D shape {vol.shape}"
    z, h, w = vol.shape
    if z < 4:
        return False, f"Volume has only {z} slices (minimum 4 required)"
    if h < 32 or w < 32:
        return False, f"Spatial dimensions too small: {h}x{w} (minimum 32x32)"
    if h > 1024 or w > 1024:
        return False, f"Spatial dimensions too large: {h}x{w} (maximum 1024x1024)"
    if not np.isfinite(vol).all():
        return False, "Volume contains NaN or Inf values"
    return True, "OK"


def safe_extract_zip(uploaded_file) -> list[bytes]:
    """Safely extract DICOM files from a ZIP with size checks."""
    total_size = uploaded_file.size
    if total_size > MAX_ZIP_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"ZIP file too large ({total_size / 1e6:.0f} MB, max {MAX_ZIP_FILE_SIZE_MB} MB)")

    bytes_list = []
    extracted_total = 0

    with zipfile.ZipFile(uploaded_file) as zf:
        # Check for zip bomb: sum of uncompressed sizes
        total_uncompressed = sum(info.file_size for info in zf.infolist())
        if total_uncompressed > MAX_ZIP_EXTRACTED_SIZE_MB * 1024 * 1024:
            raise ValueError(
                f"Extracted size would be {total_uncompressed / 1e6:.0f} MB "
                f"(max {MAX_ZIP_EXTRACTED_SIZE_MB} MB). Possible zip bomb."
            )

        dcm_names = [
            n for n in zf.namelist()
            if n.lower().endswith(".dcm") and not n.startswith("__MACOSX")
        ]

        if len(dcm_names) > MAX_DICOM_SLICES:
            raise ValueError(f"Too many DICOM files ({len(dcm_names)}, max {MAX_DICOM_SLICES})")

        for name in dcm_names:
            data = zf.read(name)
            extracted_total += len(data)
            if extracted_total > MAX_ZIP_EXTRACTED_SIZE_MB * 1024 * 1024:
                raise ValueError("Extracted data exceeds size limit")
            bytes_list.append(data)

    return bytes_list


# =========================
# DICOM processing
# =========================
def load_dicom_series_from_bytes(dicom_bytes_list: list[bytes]) -> np.ndarray:
    """Parse DICOM bytes, sort by InstanceNumber, convert to HU volume."""
    dcm_list = []
    errors = []
    for i, b in enumerate(dicom_bytes_list):
        try:
            dcm = pydicom.dcmread(io.BytesIO(b), force=True)
            if not hasattr(dcm, "pixel_array"):
                errors.append(f"File {i}: no pixel data")
                continue
            dcm_list.append(dcm)
        except Exception as e:
            errors.append(f"File {i}: {e}")

    if errors and len(errors) <= 5:
        for err in errors:
            st.warning(f"Skipped: {err}")
    elif errors:
        st.warning(f"Skipped {len(errors)} invalid DICOM files")

    if not dcm_list:
        raise ValueError("No valid DICOM files found in the ZIP archive")

    if len(dcm_list) < 4:
        raise ValueError(f"Only {len(dcm_list)} valid slices found (minimum 4 required)")

    def sort_key(d):
        return int(getattr(d, "InstanceNumber", 0))
    dcm_list.sort(key=sort_key)

    vol = np.stack(
        [d.pixel_array.astype(np.int16) for d in dcm_list], axis=0
    ).astype(np.float32)

    slope = float(getattr(dcm_list[0], "RescaleSlope", 1.0))
    inter = float(getattr(dcm_list[0], "RescaleIntercept", 0.0))
    return vol * slope + inter


# =========================
# Image processing
# =========================
def window01(hu: np.ndarray, wl: float = -175, ww: float = 425) -> np.ndarray:
    """Apply HU windowing and normalize to [0, 1]."""
    lo = wl - ww / 2
    hi = wl + ww / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)


def body_bbox(x01: np.ndarray, thr: float = 0.05):
    """Find bounding box of the body region."""
    m = x01 > thr
    if not m.any():
        return 0, x01.shape[0], 0, x01.shape[1], 0, x01.shape[2]
    zz = np.where(m.any(axis=(1, 2)))[0]
    yy = np.where(m.any(axis=(0, 2)))[0]
    xx = np.where(m.any(axis=(0, 1)))[0]
    return zz.min(), zz.max() + 1, yy.min(), yy.max() + 1, xx.min(), xx.max() + 1


def crop_resize_to_target(x01: np.ndarray, target=TARGET_SHAPE) -> np.ndarray:
    """Crop to body region and resize to target shape."""
    z0, z1, y0, y1, x0, x1 = body_bbox(x01)
    cropped = x01[z0:z1, y0:y1, x0:x1]
    if cropped.size == 0:
        st.warning("Body detection failed, using full volume")
        cropped = x01
    tz, th, tw = target
    return zoom(
        cropped,
        (tz / max(cropped.shape[0], 1), th / max(cropped.shape[1], 1), tw / max(cropped.shape[2], 1)),
        order=1,
    ).astype(np.float32)


def volume_to_sequence(
    vol01: np.ndarray, num_steps: int = DEFAULT_NUM_STEPS,
    num_slices_per_step: int = DEFAULT_NUM_SLICES_PER_STEP
) -> torch.Tensor:
    """Convert 3D volume to 2.5D sequence tensor (1, T, 3, H, W)."""
    z, h, w = vol01.shape
    centers = np.linspace(0, z - 1, num_steps, dtype=int)
    half = num_slices_per_step // 2
    frames: list[np.ndarray] = []

    for c in centers:
        start = max(c - half, 0)
        end = min(c + half + 1, z)
        slc = vol01[start:end]

        if slc.shape[0] < num_slices_per_step:
            pad_pre = half - (c - start)
            pad_post = (c + half) - (end - 1)
            pre = np.repeat(vol01[[start]], max(pad_pre, 0), axis=0) if pad_pre > 0 else np.empty((0, h, w))
            post = np.repeat(vol01[[end - 1]], max(pad_post, 0), axis=0) if pad_post > 0 else np.empty((0, h, w))
            slc = np.concatenate([pre, slc, post], axis=0)
            if slc.shape[0] > num_slices_per_step:
                slc = slc[:num_slices_per_step]
        else:
            slc = slc[:num_slices_per_step]

        if num_slices_per_step == 3:
            slc3 = slc
        elif num_slices_per_step == 1:
            slc3 = np.repeat(slc, 3, axis=0)
        else:
            idx = np.linspace(0, slc.shape[0] - 1, 3).round().astype(int)
            slc3 = slc[idx]

        frames.append(slc3)

    seq = np.stack(frames, axis=0).astype(np.float32)
    return torch.from_numpy(seq).unsqueeze(0)


def image_to_demo_sequence(
    img: Image.Image, num_steps: int = DEFAULT_NUM_STEPS,
    num_slices_per_step: int = DEFAULT_NUM_SLICES_PER_STEP
) -> torch.Tensor:
    """Convert a single image to a dummy sequence for demo purposes."""
    g = img.convert("L").resize((256, 256))
    arr = np.array(g, dtype=np.float32)
    ptp = arr.max() - arr.min()
    arr = (arr - arr.min()) / (ptp + 1e-6)
    frame = np.stack([arr] * 3, axis=0)
    seq = np.stack([frame] * num_steps, axis=0).astype(np.float32)
    return torch.from_numpy(seq).unsqueeze(0)


# =========================
# Inference
# =========================
def predict_prob_from_seq(model, seq: torch.Tensor) -> float:
    """Run inference and return probability [0, 1]."""
    seq = seq.to(DEVICE)
    with torch.no_grad():
        logit = model(seq)
        prob = torch.sigmoid(logit).item()
    return float(prob)


def risk_bucket(prob: float, thresh: float) -> tuple[str, str, str]:
    """Categorize probability into risk level with color and Thai description."""
    if prob >= thresh:
        return "HIGH RISK", "#ff0000", "ความเสี่ยงสูง - แนะนำให้แพทย์ตรวจสอบเพิ่มเติมโดยเร่งด่วน"
    if prob >= 0.5:
        return "MEDIUM RISK", "#ffa500", "ความเสี่ยงปานกลาง - แนะนำให้แพทย์ตรวจสอบเพิ่มเติม"
    return "LOW RISK", "#00ff00", "ความเสี่ยงต่ำ - ไม่พบสัญญาณผิดปกติชัดเจน"


# =========================
# Saliency / Explainability
# =========================
def compute_saliency_map(model: torch.nn.Module, seq: torch.Tensor) -> np.ndarray:
    """Compute gradient-based saliency map."""
    inp = seq.to(DEVICE).clone().detach().requires_grad_(True)
    model.zero_grad()
    prob = torch.sigmoid(model(inp))
    prob.backward()

    if inp.grad is None:
        return np.zeros((seq.shape[1], seq.shape[3], seq.shape[4]), dtype=np.float32)

    sal = inp.grad.detach().abs().cpu().numpy()[0]  # (T, 3, H, W)
    sal = np.nan_to_num(sal, nan=0.0, posinf=0.0, neginf=0.0)
    sal_per_slice = sal.sum(axis=1)  # (T, H, W)
    sal_max = sal_per_slice.max()
    if sal_max > 0:
        sal_per_slice = sal_per_slice / sal_max
    return sal_per_slice


def describe_saliency_location(saliency_map: np.ndarray, percentile: float = 99.0, border: int = 10) -> str:
    """Describe the spatial location of the highest saliency region."""
    if saliency_map.size == 0:
        return "unable to determine location"

    agg = saliency_map.mean(axis=0)
    if not np.isfinite(agg).all() or agg.max() <= 0:
        return "unable to determine location"

    H, W = agg.shape
    agg2 = agg.copy()
    if border > 0 and (H > 2 * border) and (W > 2 * border):
        agg2[:border, :] = 0
        agg2[-border:, :] = 0
        agg2[:, :border] = 0
        agg2[:, -border:] = 0

    if agg2.max() <= 0:
        return "unable to determine location"

    thr = np.percentile(agg2[agg2 > 0], percentile) if (agg2 > 0).any() else agg2.max()
    mask = agg2 >= thr
    if not mask.any():
        return "unable to determine location"

    weights = agg2[mask]
    coords = np.argwhere(mask)
    row_mean = (coords[:, 0] * weights).sum() / (weights.sum() + 1e-8)
    col_mean = (coords[:, 1] * weights).sum() / (weights.sum() + 1e-8)

    v_pos = "upper" if row_mean < H / 3 else ("central" if row_mean < 2 * H / 3 else "lower")
    h_pos = "left" if col_mean < W / 3 else ("center" if col_mean < 2 * W / 3 else "right")

    return f"{v_pos}-{h_pos}"


def vis_boost(sal: np.ndarray, p_low: float = 70.0, p_high: float = 99.7, gamma: float = 0.4) -> np.ndarray:
    """Enhance saliency contrast for visualization."""
    sal = np.maximum(sal, 0.0)
    sal = np.nan_to_num(sal, nan=0.0, posinf=0.0, neginf=0.0)
    if sal.max() <= 1e-8:
        return np.zeros_like(sal, dtype=np.float32)

    lo = np.percentile(sal, p_low)
    hi = np.percentile(sal, p_high)
    if hi <= lo + 1e-8:
        hi = float(sal.max())

    sal = np.clip(sal, lo, hi)
    sal = (sal - lo) / (hi - lo + 1e-8)
    sal = sal ** gamma
    return np.clip(sal, 0.0, 1.0).astype(np.float32)


def apply_heatmap_overlay(base_slice: np.ndarray, saliency: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Apply a proper red-yellow heatmap overlay on a CT slice."""
    base = np.clip(base_slice.astype(np.float32), 0.0, 1.0)
    rgb = np.stack([base, base, base], axis=-1)

    sal = vis_boost(saliency)

    # Red-yellow heatmap: low saliency = transparent, high = red/yellow
    heat_r = np.clip(sal * 2.0, 0.0, 1.0)
    heat_g = np.clip(sal * 2.0 - 0.5, 0.0, 1.0) * 0.7
    heat_b = np.zeros_like(sal)
    heatmap = np.stack([heat_r, heat_g, heat_b], axis=-1)

    # Blend only where saliency is significant
    mask = sal[..., np.newaxis] > 0.05
    blended = np.where(mask, (1 - alpha) * rgb + alpha * heatmap, rgb)
    return np.clip(blended, 0.0, 1.0)


def show_saliency_block(model: torch.nn.Module, seq: torch.Tensor, volume01: np.ndarray | None) -> None:
    """Display saliency map analysis section."""
    st.subheader("Saliency / Localization (Explainability)")
    st.caption(
        "Gradient-based saliency shows regions the model focused on. "
        "This is NOT a direct lesion localization - interpret with clinical context."
    )

    with st.spinner("Computing saliency map..."):
        saliency = compute_saliency_map(model, seq)

    T, H, W = saliency.shape
    t_mid = T // 2

    colA, colB = st.columns(2)
    with colA:
        sal_boost = vis_boost(saliency[t_mid])
        st.image(sal_boost, caption=f"Saliency heatmap (step {t_mid})", clamp=True)
    with colB:
        if isinstance(volume01, np.ndarray) and volume01.ndim == 3:
            z = volume01.shape[0]
            z_idx = int(np.clip(round((t_mid / max(T - 1, 1)) * (z - 1)), 0, z - 1))
            overlay = apply_heatmap_overlay(volume01[z_idx], saliency[t_mid])
            st.image(overlay, caption=f"CT + saliency overlay (z={z_idx})")

            try:
                buf = io.BytesIO()
                Image.fromarray((overlay * 255).astype(np.uint8)).save(buf, format="PNG")
                st.download_button(
                    label="Download overlay image",
                    data=buf.getvalue(),
                    file_name=f"saliency_overlay_z{z_idx}.png",
                    mime="image/png",
                )
            except Exception:
                pass

    loc_desc = describe_saliency_location(saliency)
    st.info(f"Model attention focused on: **{loc_desc}** region")


# =========================
# Result display
# =========================
def show_prediction_result(prob: float, threshold: float):
    """Display prediction results with risk level."""
    label, bg_color, description = risk_bucket(prob, threshold)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability", f"{prob:.4f}")
    with col2:
        text_color = "#000" if label != "LOW RISK" else "#000"
        st.markdown(
            f'<div style="background-color:{bg_color}; border-radius:15px; padding:20px; '
            f'color:{text_color}; font-weight:800; text-align:center; font-size:1.2rem;">'
            f'Prediction: {label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"**Assessment:** {description}")

    st.markdown(
        '<div style="background:rgba(255,200,0,0.15); border-left:4px solid #ffa500; '
        'padding:12px; border-radius:8px; margin-top:10px;">'
        '<strong>Disclaimer:</strong> This result is from an AI prototype for educational purposes only. '
        'It must NOT be used for clinical diagnosis. Always consult a qualified physician.'
        '</div>',
        unsafe_allow_html=True,
    )


def generate_report_text(prob: float, threshold: float, input_type: str, saliency_loc: str = "") -> str:
    """Generate a text report of the assessment."""
    label, _, description = risk_bucket(prob, threshold)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""{'=' * 60}
CT BOWEL INJURY RISK ASSESSMENT REPORT
{'=' * 60}

Date/Time:       {now}
Input Type:      {input_type}
Model:           ResNet18 + GRU (2.5D CNN)
Threshold:       {threshold:.2f}
Device:          {DEVICE.upper()}

{'=' * 60}
RESULTS
{'=' * 60}

Risk Probability:   {prob:.4f} ({prob*100:.2f}%)
Risk Level:         {label}
Assessment:         {description}
"""
    if saliency_loc:
        report += f"Attention Region:   {saliency_loc}\n"

    report += f"""
{'=' * 60}
DISCLAIMER
{'=' * 60}

This report is generated by an AI prototype for EDUCATIONAL
PURPOSES ONLY. It is NOT intended for clinical diagnosis or
medical decision-making. The results should be interpreted
by qualified medical professionals only.

Model trained on RSNA 2023 Abdominal Trauma Detection dataset.
Performance: Accuracy 96.09%, ROC-AUC 93.78% (validation set).

{'=' * 60}
"""
    return report


# =========================
# CSS Theme
# =========================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at top left, #1f2937 0%, #0f172a 100%) !important;
  font-family: 'Poppins', sans-serif !important;
  color: #e5e7eb !important;
}

html, body, [data-testid="stAppViewContainer"] * {
  color: #e5e7eb;
  opacity: 1 !important;
  filter: none !important;
  text-shadow: none !important;
}

h1, h2, h3 {
  background: -webkit-linear-gradient(315deg, #5eead4 0%, #22d3ee 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700 !important;
  margin-bottom: 0.5rem !important;
}

button {
  background-image: linear-gradient(145deg, #06b6d4, #3b82f6) !important;
  color: #ffffff !important;
  border-radius: 30px !important;
  padding: 14px 28px !important;
  border: none !important;
  font-weight: 600 !important;
  box-shadow: 0 10px 20px rgba(3, 169, 244, 0.25), 0 6px 6px rgba(0, 0, 0, 0.1) !important;
  transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}
button:hover {
  transform: translateY(-3px) scale(1.02) !important;
  box-shadow: 0 14px 24px rgba(3, 169, 244, 0.35), 0 8px 8px rgba(0, 0, 0, 0.1) !important;
}

.stMetric, div.element-container, .stExpander {
  background: rgba(255, 255, 255, 0.05) !important;
  backdrop-filter: blur(10px) !important;
  border-radius: 20px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25) !important;
}

section[data-testid="stSidebar"] {
  background-color: rgba(17, 24, 39, 0.92) !important;
  border-right: 1px solid rgba(255,255,255,0.12) !important;
}

section[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
  opacity: 1 !important;
  filter: none !important;
}

.stRadio *, .stSelectbox *, .stNumberInput *, .stSlider *, .stTextInput *, .stTextArea * {
  color: #e5e7eb !important;
  opacity: 1 !important;
}

.stRadio > div, .stSelectbox > div > div {
  background-color: rgba(255,255,255,0.06) !important;
  border-radius: 12px !important;
  padding: 12px !important;
}

.stFileUploader {
  border: 2px dashed #38bdf8 !important;
  border-radius: 20px !important;
  padding: 30px !important;
  background-color: rgba(30, 41, 59, 0.92) !important;
}

[data-testid="stFileUploaderDropzone"] {
  background-color: rgba(30, 41, 59, 0.94) !important;
  border-radius: 20px !important;
  opacity: 1 !important;
}

.stFileUploader *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploaderDropzone"] path,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] div {
  color: #e5e7eb !important;
  opacity: 1 !important;
  filter: none !important;
}

.stFileUploader:hover { border-color: #7dd3fc !important; }

.stImage {
  border-radius: 20px !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
  transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}
.stImage:hover {
  transform: scale(1.02) !important;
  box-shadow: 0 14px 40px rgba(0,0,0,0.4) !important;
}

.stTabs * { opacity: 1 !important; color: #e5e7eb !important; }
.stCaption, .stMarkdown, .stMarkdown * { opacity: 1 !important; }
.stAlert p { color: #000000 !important; font-weight: 700 !important; }
</style>
"""


# =========================
# Page: Gallery
# =========================
def page_gallery():
    st.header("Gallery")
    imgs = st.file_uploader(
        "Upload images (JPG/PNG, multiple files)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if imgs:
        cols = st.columns(3)
        for i, f in enumerate(imgs[:12]):
            cols[i % 3].image(f)


# =========================
# Page: AI Prediction
# =========================
def page_prediction(model, threshold, num_steps, num_slices):
    st.header("AI Prediction")

    tab1, tab2 = st.tabs([".npy Volume", "ZIP DICOM"])

    with tab1:
        st.markdown("Upload a preprocessed `.npy` volume with shape (Z, H, W), values in [0, 1].")
        up = st.file_uploader("Upload .npy file", type="npy", key="npy_upload")
        if up:
            # File size check
            if up.size > MAX_NPY_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File too large ({up.size / 1e6:.0f} MB, max {MAX_NPY_FILE_SIZE_MB} MB)")
                return

            progress = st.progress(0, text="Loading volume...")
            try:
                vol = np.load(up)
            except Exception as e:
                st.error(f"Failed to load .npy file: {e}")
                return

            # Validate
            valid, msg = validate_npy_volume(vol)
            if not valid:
                st.error(f"Invalid volume: {msg}")
                return

            vol01 = np.clip(vol.astype(np.float32), 0, 1)
            st.caption(f"Volume shape: {vol01.shape}, dtype: {vol.dtype}")

            mid = vol01.shape[0] // 2
            st.image(vol01[mid], "Preview Slice (middle)", clamp=True)
            progress.progress(40, text="Creating sequence...")

            seq = volume_to_sequence(vol01, num_steps=num_steps, num_slices_per_step=num_slices)
            progress.progress(70, text="Running inference...")

            prob = predict_prob_from_seq(model, seq)
            progress.progress(100, text="Done!")

            show_prediction_result(prob, threshold)

            # Report download
            report = generate_report_text(prob, threshold, f".npy ({vol01.shape})")
            st.download_button(
                "Download Report (TXT)",
                data=report,
                file_name=f"bowel_injury_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt",
                mime="text/plain",
            )

            if st.checkbox("Show saliency map (explainability)", key="sal_npy"):
                show_saliency_block(model, seq, volume01=vol01)

    with tab2:
        st.markdown("Upload a ZIP file containing DICOM (.dcm) slices from a CT scan.")
        up = st.file_uploader("Upload ZIP with DICOM slices", type="zip", key="zip_upload")
        if up:
            progress = st.progress(0, text="Extracting DICOM files...")

            try:
                bytes_list = safe_extract_zip(up)
            except ValueError as e:
                st.error(str(e))
                return

            if not bytes_list:
                st.error("No .dcm files found in the ZIP archive")
                return

            st.caption(f"Found {len(bytes_list)} DICOM slices")
            progress.progress(20, text="Parsing DICOM...")

            try:
                hu = load_dicom_series_from_bytes(bytes_list)
            except ValueError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Failed to process DICOM: {e}")
                return

            progress.progress(40, text="Windowing and cropping...")
            vol01 = window01(hu)
            vol = crop_resize_to_target(vol01)
            progress.progress(60, text="Creating sequence...")

            mid = vol.shape[0] // 2
            st.image(vol[mid].astype(np.float32), "Processed Preview (middle slice)", clamp=True)

            seq = volume_to_sequence(vol.astype(np.float32), num_steps=num_steps, num_slices_per_step=num_slices)
            progress.progress(80, text="Running inference...")

            prob = predict_prob_from_seq(model, seq)
            progress.progress(100, text="Done!")

            show_prediction_result(prob, threshold)

            report = generate_report_text(prob, threshold, f"DICOM ZIP ({len(bytes_list)} slices)")
            st.download_button(
                "Download Report (TXT)",
                data=report,
                file_name=f"bowel_injury_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt",
                mime="text/plain",
            )

            if st.checkbox("Show saliency map (explainability)", key="sal_zip"):
                show_saliency_block(model, seq, volume01=vol.astype(np.float32))


# =========================
# Page: Demo
# =========================
def page_demo(model, threshold, num_steps, num_slices):
    st.header("Demo Mode (JPEG/PNG)")

    st.warning(
        "**DEMO ONLY** - This mode uses a single 2D image duplicated to simulate a 3D volume. "
        "Results are NOT clinically meaningful. Use the AI Prediction tab with real CT data "
        "for actual assessments."
    )

    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="demo_upload")
    if up:
        img = Image.open(up)
        st.image(img, caption="Uploaded image")

        seq = image_to_demo_sequence(img, num_steps=num_steps, num_slices_per_step=num_slices)
        prob = predict_prob_from_seq(model, seq)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Demo Probability", f"{prob:.4f}")
        with col2:
            label, bg_color, _ = risk_bucket(prob, threshold)
            st.markdown(
                f'<div style="background-color:{bg_color}; border-radius:15px; padding:20px; '
                f'color:#000; font-weight:800; text-align:center; font-size:1.2rem;">'
                f'DEMO: {label}</div>',
                unsafe_allow_html=True,
            )

        st.error(
            "This result is from a DEMO using a single image, NOT real CT data. "
            "Do not interpret this as a valid medical assessment."
        )


# =========================
# Page: About
# =========================
def page_about():
    st.header("About This Project")

    st.markdown("""
### AI-Based Risk Assessment of Bowel Injury from Abdominal CT

This prototype was developed as an educational project to demonstrate how deep learning
can assist in screening for bowel injuries from abdominal CT scans.

**Architecture:**
- **Feature Extraction:** ResNet18 (pretrained on ImageNet) extracts spatial features from each CT slice
- **Temporal Modeling:** GRU (Gated Recurrent Unit) learns patterns across the sequence of slices
- **Classification:** Final fully-connected layer outputs injury probability (0-1)

**Data Source:** RSNA 2023 Abdominal Trauma Detection AI Challenge
- 4,000+ cases from 23 sites in 14 countries
- Binary classification: bowel injury present/absent

**Performance (Validation Set, threshold=0.7):**

| Metric | Value |
|--------|-------|
| Accuracy | 96.09% |
| Precision | 78.57% |
| Recall | 84.62% |
| F1-score | 81.48% |
| ROC-AUC | 93.78% |

**Limitations:**
- No pixel-level lesion annotation
- Limited dataset diversity
- Gradient-based saliency may not accurately localize lesions
- Not validated in clinical settings

**Team:** Chakireen Asae, Abdulkoffar Nuidam
**Advisor:** Arfan Baka
**School:** Princess Chulabhorn Science High School Satun

---
*This system is designed to support medical decision-making, NOT replace physician diagnosis.*
    """)


# =========================
# Main App
# =========================
st.set_page_config(
    page_title="CT Bowel Injury Detection",
    layout="wide",
    page_icon="🩺",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("CT Bowel Injury Detection")
st.markdown(
    "<div style='font-size:1.05rem; color:#e5e7eb; margin-bottom:1.5rem; opacity:1;'>"
    "AI-powered risk assessment for bowel injury from abdominal CT scans"
    "</div>",
    unsafe_allow_html=True,
)
st.error(
    "This is an educational prototype only. NOT for clinical diagnosis. "
    "Always consult a qualified physician for medical decisions."
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("HIGH RISK threshold", 0.5, 0.9, 0.7, 0.05)
    num_steps_input = st.number_input(
        "Number of steps", min_value=8, max_value=96,
        value=DEFAULT_NUM_STEPS, step=2
    )
    num_slices_input = st.radio(
        "Slices per step",
        options=[1, 3, 5],
        index=[1, 3, 5].index(DEFAULT_NUM_SLICES_PER_STEP),
    )

    st.markdown("---")
    st.caption(f"Device: **{DEVICE.upper()}**")
    st.caption(f"PyTorch: {torch.__version__}")
    st.caption(f"Target shape: {TARGET_SHAPE}")

    page = st.radio(
        "Menu",
        ["AI Prediction", "Gallery", "Demo (Single Image)", "About"],
    )

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info(
        "Please check your internet connection or manually place the model file "
        f"'{MODEL_PATH}' in the application directory."
    )
    st.stop()

# Route to pages
num_steps = int(num_steps_input)
num_slices = int(num_slices_input)

if page == "AI Prediction":
    page_prediction(model, threshold, num_steps, num_slices)
elif page == "Gallery":
    page_gallery()
elif page == "Demo (Single Image)":
    page_demo(model, threshold, num_steps, num_slices)
elif page == "About":
    page_about()

# Footer
st.markdown("---")
st.caption(
    "Educational prototype | Developed by REEN (PCCST Satun) | "
    "NOT for medical use | Based on RSNA 2023 data"
)
