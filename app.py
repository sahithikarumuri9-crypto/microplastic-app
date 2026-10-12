
import streamlit as st, numpy as np, cv2
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).parent
st.set_page_config(page_title="Microplastic detector", layout="wide")
st.title("Microplastic detection and counting")

@st.cache_resource
def load(name): return YOLO(str(HERE / "models" / name))

mode = st.radio("Model", ["Detection", "Segmentation"], horizontal=True)
imgsz = st.select_slider("Inference size (smaller = faster)", [320, 480, 640, 800, 1024], value=640)
conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if up:
    img = np.array(Image.open(up).convert("RGB"))
    model = load("det_best.pt" if mode == "Detection" else "seg_best.pt")
    r = model.predict(img[..., ::-1], imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    st.image(r.plot()[..., ::-1], caption=f"{len(r.boxes)} particles found", use_container_width=True)
    names = [model.names[int(c)] for c in r.boxes.cls]
    if names:
        st.bar_chart(
            {n: names.count(n) for n in sorted(set(names))}
        )
