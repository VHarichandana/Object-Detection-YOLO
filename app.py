"""

Streamlit app that detects objects in an uploaded image using a pretrained
Ultralytics YOLO model, displays detections, and lets the user download results.
"""
from pathlib import Path
from io import BytesIO
import csv

import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="YOLO Object Detection", page_icon="🔎", layout="wide")
st.title("🔎 Object Detection with YOLO")
st.caption("CODTECH AI Internship")
st.write(
    "Upload an image to detect supported objects. The pretrained model returns "
    "class labels, bounding boxes, and confidence scores."
)

@st.cache_resource
def load_model():
    # The model weights download automatically on the first run.
    return YOLO("yolo11n.pt")

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Minimum confidence", min_value=0.10, max_value=0.90,
                           value=0.25, step=0.05)
    st.info("Lower confidence can show more detections, including false positives.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a clear image containing objects such as people, vehicles, or animals.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        left, right = st.columns(2)
        with left:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

        with st.spinner("Loading YOLO and detecting objects..."):
            model = load_model()
            results = model.predict(source=image, conf=confidence, verbose=False)
            result = results[0]
            annotated_bgr = result.plot()
            # result.plot() returns a BGR NumPy image; convert it to RGB for Pillow.
            import cv2
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(annotated_rgb)

        with right:
            st.subheader("Detected objects")
            st.image(annotated_image, use_container_width=True)

        # Extract prediction details into a table and CSV download.
        rows = []
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                rows.append({
                    "class_id": class_id,
                    "label": names.get(class_id, str(class_id)),
                    "confidence": round(score, 4),
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                })

        st.subheader("Detection summary")
        metric1, metric2 = st.columns(2)
        metric1.metric("Objects detected", len(rows))
        metric2.metric("Unique classes", len({row["label"] for row in rows}))

        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.warning("No objects met the selected confidence threshold. Try lowering it.")

        # Encode annotated image in memory for the download button.
        image_buffer = BytesIO()
        annotated_image.save(image_buffer, format="PNG")
        st.download_button(
            "Download annotated image",
            data=image_buffer.getvalue(),
            file_name="yolo_detections.png",
            mime="image/png",
        )

        csv_buffer = BytesIO()
        # UTF-8 CSV for spreadsheet compatibility.
        import pandas as pd
        pd.DataFrame(rows, columns=[
            "class_id", "label", "confidence", "x_min", "y_min", "x_max", "y_max"
        ]).to_csv(csv_buffer, index=False, encoding="utf-8")
        st.download_button(
            "Download detections as CSV",
            data=csv_buffer.getvalue(),
            file_name="detections.csv",
            mime="text/csv",
        )
        st.caption("Note: predictions depend on image quality, model classes, and confidence threshold.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
        st.info("Check the installation steps in README.md and ensure the model can download on first run.")
else:
    st.info("Upload an image above to begin.")
