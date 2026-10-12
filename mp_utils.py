"""Helper functions for the Microplastic AI Analyzer (detection + segmentation + geometry + calibration).

Nothing here retrains a model. Weights are loaded from explicit paths (see load_config()).
Physical sizes (um) are produced ONLY when a valid um-per-pixel value is supplied.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PALETTE = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # RGB

DEFAULT_CONFIG = {
    "detector_weights": "models/best.pt",       # YOLOv8 polymer detector (required)
    "segmenter_weights": "models/seg_best.pt",  # YOLOv8-seg particle segmenter (optional)
    "imgsz": 1024,
    "max_upload_mb": 25,
    "max_megapixels": 40,
    "calibration_profiles": {},                 # e.g. {"scope_A_10x": {"um_per_pixel": 0.8, "note": "..."}}
}


# ----------------------------------------------------------------------------- config
def load_config():
    """config.json (next to app.py) overrides defaults; environment variables override both."""
    cfg = dict(DEFAULT_CONFIG)
    p = APP_DIR / "config.json"
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    cfg["detector_weights"] = os.environ.get("MP_DET_WEIGHTS", cfg["detector_weights"])
    cfg["segmenter_weights"] = os.environ.get("MP_SEG_WEIGHTS", cfg["segmenter_weights"])
    for key in ("detector_weights", "segmenter_weights"):       # relative paths are relative to the app folder
        q = Path(cfg[key])
        cfg[key] = str(q if q.is_absolute() else APP_DIR / q)
    return cfg


# ----------------------------------------------------------------------------- calibration
def um_per_pixel_from_scale_bar(known_length_um, bar_length_px):
    """um_per_pixel = known_length_um / scale_bar_length_pixels (both must be > 0)."""
    if known_length_um is None or bar_length_px is None or known_length_um <= 0 or bar_length_px <= 0:
        raise ValueError("Scale-bar length (um) and its length in pixels must both be positive numbers.")
    return float(known_length_um) / float(bar_length_px)


def rescale_calibration(um_per_px, calibrated_width_px, analysed_width_px):
    """If the scale was measured on an image of one width but analysis runs on another width
    (resized / cropped copy), convert the calibration. Cropping does not change um/px; resizing does."""
    return float(um_per_px) * float(calibrated_width_px) / float(analysed_width_px)


def valid_um_per_px(v):
    return v is not None and np.isfinite(v) and v > 0


# ----------------------------------------------------------------------------- geometry
def geometry_from_contour(contour):
    """contour: Nx2 or Nx1x2 array of pixel coordinates. Returns pixel-unit geometry or None.
    area  = polygon area; length/width = sides of the minimum-area rotated rectangle (long/short)."""
    c = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    if len(c) < 3:
        return None
    area = float(cv2.contourArea(c))
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    length, width = float(max(rw, rh)), float(min(rw, rh))
    if area <= 0 or width <= 0:
        return None
    return dict(area_px=area, length_px=length, width_px=width, aspect_ratio=length / width)


def otsu_contour_in_box(img_bgr, box, pad=4):
    """EXPERIMENTAL fallback: Otsu threshold inside a detection box. Not a trained segmentation."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, w), min(y2 + pad, h)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    return c + np.array([x1, y1], dtype=np.float32)


def _box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_polygons_to_boxes(boxes, polys, min_iou=0.3):
    """Greedy one-to-one match of segmentation polygons to detection boxes by bounding-box IoU.
    Returns {box_index: polygon_index}."""
    pboxes = []
    for p in polys:
        x, y, w, h = cv2.boundingRect(np.asarray(p, dtype=np.float32).reshape(-1, 1, 2))
        pboxes.append((x, y, x + w, y + h))
    pairs = sorted(((_box_iou(b, pb), i, j) for i, b in enumerate(boxes) for j, pb in enumerate(pboxes)),
                   reverse=True)
    used_b, used_p, out = set(), set(), {}
    for iou, i, j in pairs:
        if iou < min_iou:
            break
        if i in used_b or j in used_p:
            continue
        out[i] = j; used_b.add(i); used_p.add(j)
    return out


# ----------------------------------------------------------------------------- main analysis
def analyze_image(img_bgr, det_model, seg_model=None, conf=0.25, imgsz=1024, um_per_px=None,
                  seg_conf=0.25, device=None, allow_otsu_fallback=False, brighten=2.0):
    """Run detection (+ optional segmentation) and compute per-particle geometry.

    Returns (table: DataFrame, boxes_rgb: ndarray, overlay_rgb: ndarray, counts: DataFrame)."""
    names = det_model.names
    r = det_model.predict(img_bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
    confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
    cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros(0, dtype=int)

    polys = []
    if seg_model is not None:
        sr = seg_model.predict(img_bgr, imgsz=imgsz, conf=seg_conf, device=device, verbose=False)[0]
        if sr.masks is not None:
            polys = [np.asarray(p, dtype=np.float32) for p in sr.masks.xy if len(p) >= 3]
    matches = match_polygons_to_boxes(boxes, polys) if len(boxes) and polys else {}

    calibrated = valid_um_per_px(um_per_px)
    rows, contours = [], []
    for i, (b, cf, c) in enumerate(zip(boxes, confs, cls)):
        status, contour = "unavailable", None
        if i in matches:
            contour, status = polys[matches[i]], "segmentation_model"
        elif allow_otsu_fallback:
            contour = otsu_contour_in_box(img_bgr, b)
            status = "otsu_fallback_experimental" if contour is not None else "unavailable"
        g = geometry_from_contour(contour) if contour is not None else None
        if g is None and status != "unavailable":
            status = "unavailable"; contour = None
        row = dict(particle_id=i + 1, polymer=names[int(c)], confidence=round(float(cf), 4),
                   x1=round(float(b[0]), 1), y1=round(float(b[1]), 1),
                   x2=round(float(b[2]), 1), y2=round(float(b[3]), 1), segmentation_status=status,
                   area_px=np.nan, length_px=np.nan, width_px=np.nan, aspect_ratio=np.nan,
                   area_um2=np.nan, length_um=np.nan, width_um=np.nan)
        if g:
            row.update({k: round(v, 3) for k, v in g.items()})
            if calibrated:
                row["area_um2"] = round(g["area_px"] * um_per_px ** 2, 3)
                row["length_um"] = round(g["length_px"] * um_per_px, 3)
                row["width_um"] = round(g["width_px"] * um_per_px, 3)
        rows.append(row); contours.append(contour)
    table = pd.DataFrame(rows, columns=["particle_id", "polymer", "confidence", "x1", "y1", "x2", "y2",
                                        "segmentation_status", "area_px", "length_px", "width_px",
                                        "aspect_ratio", "area_um2", "length_um", "width_um"])
    counts = pd.DataFrame({"polymer": [names[k] for k in sorted(names)]})
    counts["count"] = [int((table.polymer == n).sum()) for n in counts.polymer]

    base = np.clip(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) * brighten, 0, 255).astype(np.uint8)
    boxes_rgb, overlay = base.copy(), base.copy()
    thick = max(1, int(round(max(base.shape[:2]) / 700)))
    for row, b, c, contour in zip(rows, boxes, cls, contours):
        col = PALETTE[int(c) % len(PALETTE)]
        p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
        cv2.rectangle(boxes_rgb, p1, p2, col, thick)
        cv2.putText(boxes_rgb, f"{row['polymer']} {row['confidence']:.2f}", (p1[0], max(p1[1] - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick + 0.2, col, 1, cv2.LINE_AA)
        if contour is not None:
            cv2.polylines(overlay, [np.round(contour).astype(np.int32).reshape(-1, 1, 2)], True, col, thick)
    return table, boxes_rgb, overlay, counts


def load_image_bgr(file_like_or_path):
    """Read an uploaded image (JPG/PNG) into a BGR uint8 array. Raises ValueError if unreadable."""
    from PIL import Image
    try:
        im = Image.open(file_like_or_path)
        im.load()
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Could not read the image: {e}") from e
