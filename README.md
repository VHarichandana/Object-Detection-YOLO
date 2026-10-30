# Object Detection (YOLO)

## Project Scope

This project provides a web interface for detecting objects in uploaded images with a pretrained Ultralytics YOLO model. It displays an annotated image with bounding boxes, class labels, and confidence scores. Users can download the annotated image and a CSV containing the detection coordinates.

This implementation uses pretrained weights for inference; it does not claim to train a custom model.

## Features

- Upload JPG, JPEG, PNG, or WEBP images.
- Adjust the minimum detection confidence.
- View original and annotated images side by side.
- Review detected classes, confidence scores, and bounding-box coordinates.
- Download the annotated image as PNG and predictions as CSV.

## Technology Stack

- Python
- Ultralytics YOLO (`yolo11n.pt`)
- Streamlit
- OpenCV
- Pillow and Pandas

## Project Structure

```text
codtech-yolo-object-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── tests/
│   └── test_project_files.py
└── screenshots/
    └── README.md
```

## Setup and Run (Windows)

1. Install Python 3.10 or 3.11 from https://www.python.org/downloads/ and enable **Add Python to PATH** during installation.
2. Open a terminal in this project folder.
3. Create a virtual environment:

   ```bash
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

4. Upgrade pip and install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed in the terminal, usually `http://localhost:8501`.
7. Upload a test image. On first run, the YOLO weights download automatically, so internet access is required.

## Dataset / Test Images

This is an inference application, so a training dataset is not required. Use a photo you own or have permission to use. For repeatable testing, you can also use the sample images and datasets linked from the official Ultralytics documentation:

- YOLO Python usage: https://docs.ultralytics.com/usage/python/
- Prediction mode: https://docs.ultralytics.com/modes/predict/

Do not commit private images or large model-weight files to the repository. Record the source and license of any external test images you include.

## How It Works

1. Streamlit accepts an uploaded image.
2. Pillow opens the image and converts it to RGB.
3. Ultralytics loads the pretrained YOLO weights.
4. The model predicts object classes, confidence scores, and bounding boxes.
5. The app renders the boxes and labels over the image.
6. Prediction details are shown in a table and can be downloaded as CSV.

## Testing Checklist

- [ ] App launches with `streamlit run app.py`.
- [ ] Valid JPG/PNG image is accepted.
- [ ] Annotated image appears after prediction.
- [ ] Detection table displays label, confidence, and coordinates.
- [ ] PNG and CSV downloads work.
- [ ] Low-confidence/no-detection case shows a helpful message.
- [ ] Invalid file types are rejected by the uploader.
- [ ] Screenshots are saved in `screenshots/`.

Run the lightweight project-file test with:

```bash
pytest -q
```

## Screenshots and Output Images

After running the app, capture screenshots showing:
1. The app before an image is uploaded.
2. The original image and its annotated result.
3. The detection summary table and download buttons.

Save these in `screenshots/`. Only include genuine screenshots from your own run; do not present mockups as actual results.

## Limitations

- The pretrained model detects only classes represented in its training labels.
- Detection accuracy can vary with image quality, lighting, object size, and occlusion.
- Confidence is a model score, not a guarantee that a detection is correct.
- This project is a demonstration and should not be used as the sole basis for safety-critical decisions.

## Credits

- Ultralytics documentation: https://docs.ultralytics.com/
- Streamlit documentation: https://docs.streamlit.io/

Check the current Ultralytics licensing terms before using the project commercially. This repository is intended for educational internship work.
