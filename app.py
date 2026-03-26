import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FreshHarvest Inspector",
    page_icon="🍓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0d0d0d;
        color: #f0ece4;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 2rem 2rem 4rem;
        max-width: 760px;
    }

    /* ── Hero header ── */
    .hero {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        color: #7aed8f;
        font-family: 'DM Sans', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 0.35rem 1rem;
        border-radius: 100px;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.05;
        color: #f0ece4;
        margin: 0 0 0.6rem;
        letter-spacing: -0.03em;
    }
    .hero-title span {
        color: #7aed8f;
    }
    .hero-sub {
        color: #666;
        font-size: 0.95rem;
        font-weight: 300;
        margin: 0;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2a2a2a !important;
        border-radius: 16px !important;
        background: #111 !important;
        padding: 1rem !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #7aed8f !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background: transparent !important;
    }

    /* ── Result cards ── */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid;
        text-align: center;
    }
    .result-fresh {
        background: #0a1f0e;
        border-color: #1d5c28;
    }
    .result-spoiled {
        background: #1f0a0a;
        border-color: #5c1d1d;
    }
    .result-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.4rem;
    }
    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 0.3rem;
    }
    .result-fresh .result-label  { color: #7aed8f; }
    .result-spoiled .result-label { color: #ed7a7a; }
    .result-confidence {
        font-size: 0.88rem;
        color: #555;
        font-weight: 300;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Confidence bar ── */
    .conf-bar-wrap {
        background: #1a1a1a;
        border-radius: 100px;
        height: 8px;
        margin: 0.8rem 0 0.3rem;
        overflow: hidden;
    }
    .conf-bar-fill-fresh {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #1d5c28, #7aed8f);
        transition: width 0.6s ease;
    }
    .conf-bar-fill-spoiled {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #5c1d1d, #ed7a7a);
        transition: width 0.6s ease;
    }
    .conf-pct {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #f0ece4;
        margin-top: 0.2rem;
    }

    /* ── Fruit pills ── */
    .fruits-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        justify-content: center;
        margin: 1.4rem 0 0.5rem;
    }
    .fruit-pill {
        background: #161616;
        border: 1px solid #222;
        color: #666;
        font-size: 0.78rem;
        padding: 0.28rem 0.8rem;
        border-radius: 100px;
        font-weight: 400;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid #1e1e1e;
        margin: 2rem 0;
    }

    /* ── Info strip ── */
    .info-strip {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        margin: 1.5rem 0;
    }
    .info-item {
        text-align: center;
    }
    .info-num {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #7aed8f;
    }
    .info-lbl {
        font-size: 0.75rem;
        color: #444;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Instruction card ── */
    .hint-card {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        color: #444;
        font-size: 0.85rem;
        line-height: 1.7;
        margin-top: 1rem;
    }
    .hint-card b { color: #666; }

    /* ── Button ── */
    .stButton > button {
        background: #7aed8f !important;
        color: #0d0d0d !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        letter-spacing: 0.02em !important;
        transition: opacity 0.2s !important;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MODEL_PATH    = "freshharvest_resnet50.pt"
IMAGE_SIZE    = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ["Fresh", "Spoiled"]
FRUIT_NAMES   = [
    "Banana", "Lemon", "Lulo", "Mango",
    "Orange", "Strawberry", "Tamarillo", "Tomato"
]

TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────
# MODEL LOADER  (cached — loads only once)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(256, 2),
    )
    checkpoint = torch.load(MODEL_PATH,
                            map_location=torch.device("cpu"),
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint.get("test_acc", 99.96)


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────

def predict(model, image: Image.Image):
    tensor = TRANSFORM(image).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        logits  = model(tensor)
        probs   = torch.softmax(logits, dim=1)[0]
    pred_idx    = probs.argmax().item()
    pred_label  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx].item() * 100
    fresh_prob  = probs[0].item() * 100
    spoiled_prob = probs[1].item() * 100
    return pred_label, confidence, fresh_prob, spoiled_prob


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI Freshness Inspector · FreshHarvest Logistics</div>
    <h1 class="hero-title">Is it <span>Fresh</span>?</h1>
    <p class="hero-sub">Drop any fruit image — the model will tell you instantly.</p>
</div>
""", unsafe_allow_html=True)

# ── Fruit pills ───────────────────────────────
fruit_pills = "".join(
    f'<span class="fruit-pill">{f}</span>' for f in FRUIT_NAMES
)
st.markdown(
    f'<div class="fruits-row">{fruit_pills}</div>',
    unsafe_allow_html=True,
)

# ── Stats strip ───────────────────────────────
st.markdown("""
<div class="info-strip">
    <div class="info-item">
        <div class="info-num">99.96%</div>
        <div class="info-lbl">Model accuracy</div>
    </div>
    <div class="info-item">
        <div class="info-num">8</div>
        <div class="info-lbl">Fruit types</div>
    </div>
    <div class="info-item">
        <div class="info-num">ResNet50</div>
        <div class="info-lbl">Architecture</div>
    </div>
    <div class="info-item">
        <div class="info-num">10</div>
        <div class="info-lbl">Training epochs</div>
    </div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────

with st.spinner("Loading model..."):
    try:
        model, test_acc = load_model()
        model_loaded    = True
    except FileNotFoundError:
        model_loaded = False


if not model_loaded:
    st.error(
        f"❌ Model file `{MODEL_PATH}` not found. "
        "Make sure it is in the same folder as this app.",
        icon="⚠️",
    )
    st.stop()


# ─────────────────────────────────────────────
# FILE UPLOADER  (drag & drop)
# ─────────────────────────────────────────────

uploaded = st.file_uploader(
    label="Drag & drop a fruit image here",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supports JPG, PNG, BMP and WebP formats.",
    label_visibility="visible",
)

st.markdown("""
<div class="hint-card">
    <b>How to use:</b> Drag an image directly onto the box above, or click
    <b>Browse files</b> to pick one from your computer. The model will
    classify it as <b>Fresh</b> or <b>Spoiled</b> in under a second.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Show uploaded image ────────────────────
    col_img, col_space = st.columns([1, 0.05])
    with col_img:
        st.image(image, caption=f"📁 {uploaded.name}",
                 use_container_width=True)

    # ── Run prediction ─────────────────────────
    with st.spinner("Analysing..."):
        label, confidence, fresh_pct, spoiled_pct = predict(model, image)

    is_fresh    = label == "Fresh"
    card_class  = "result-fresh" if is_fresh else "result-spoiled"
    emoji       = "✅" if is_fresh else "❌"
    bar_class   = "conf-bar-fill-fresh" if is_fresh else "conf-bar-fill-spoiled"

    # ── Result card ───────────────────────────
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">{label}</div>
        <div class="result-confidence">Confidence · {confidence:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bars ───────────────────────
    st.markdown("**Probability breakdown**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <p style="color:#7aed8f;font-size:0.82rem;margin:0 0 4px;
                  text-transform:uppercase;letter-spacing:.08em;">
            ✅ Fresh
        </p>
        <div class="conf-bar-wrap">
            <div class="conf-bar-fill-fresh"
                 style="width:{fresh_pct:.1f}%"></div>
        </div>
        <div class="conf-pct">{fresh_pct:.2f}%</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <p style="color:#ed7a7a;font-size:0.82rem;margin:0 0 4px;
                  text-transform:uppercase;letter-spacing:.08em;">
            ❌ Spoiled
        </p>
        <div class="conf-bar-wrap">
            <div class="conf-bar-fill-spoiled"
                 style="width:{spoiled_pct:.1f}%"></div>
        </div>
        <div class="conf-pct">{spoiled_pct:.2f}%</div>
        """, unsafe_allow_html=True)

    # ── Action message ────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    if is_fresh:
        st.success(
            "✅ This fruit is **fresh** and safe to dispatch. "
            "No action needed.",
            icon="🍃",
        )
    else:
        st.error(
            "⚠️ This fruit appears **spoiled**. "
            "Remove it from the conveyor belt immediately.",
            icon="🚫",
        )

    # ── Try another button ────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Inspect another fruit"):
        st.rerun()


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("""
<hr class="divider">
<p style="text-align:center;color:#2a2a2a;font-size:0.78rem;">
    FreshHarvest Logistics · AI Freshness Inspection System ·
    Powered by ResNet50 Transfer Learning
</p>
""", unsafe_allow_html=True)