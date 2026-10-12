# Microplastic AI Analyzer

Streamlit dashboard for: detection + 6-polymer classification (YOLOv8), counting, optional particle
segmentation + geometry, and calibrated size estimation (µm) when a valid scale is provided.

## Status (be honest in demos)
| Feature | Status |
|---|---|
| Detection, polymer classification, counting | Implemented (YOLOv8n trained on SOMBSUK; held-out test metrics are in the notebook) |
| CSV / annotated-image export | Implemented |
| Particle segmentation | Needs `models/seg_best.pt` (trained in the notebook on a *different* dataset - see domain-gap note) |
| Otsu fallback geometry | Experimental, unvalidated |
| µm sizes | Only with a valid calibration you supply; never assumed |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir models   # put best.pt (and optionally seg_best.pt) here
cp config.example.json config.json   # optional
streamlit run app.py
```
Weights can also come from env vars: `MP_DET_WEIGHTS=/path/best.pt MP_SEG_WEIGHTS=/path/seg_best.pt`.

## Calibration
`um_per_pixel = known_length_um / scale_bar_length_pixels`, measured with the same microscope, objective and
camera settings as the analysed image, on the original-resolution image. Different magnifications need
different values. Optional saved values go in `config.json` under `calibration_profiles`.

## Deployment
* Hugging Face Spaces (Streamlit SDK or Docker): store weights with Git LFS or in a model repo and download at
  start-up. CPU inference at 1024 px takes seconds per image.
* Streamlit Community Cloud: repo must contain the weights (small YOLOv8n files are fine) and requirements.txt;
  check its current memory limit - PyTorch is large.
* Never commit tokens or credentials. Do not bundle the training dataset.
