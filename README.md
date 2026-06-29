# Deepfake Video Detection

## Project Overview
This project develops a hybrid deepfake video detection system that combines spatial and temporal deep learning to detect manipulated facial videos. Five models are compared using different CNN backbones (MobileNetV2, ResNet50, Xception) combined with BiLSTM and an attention mechanism. The final proposed model is **ResNet50+BiLSTM+Attention**, selected for its strong performance and frame-level interpretability.

## Features
- MTCNN face detection and cropping
- ResNet50 feature extraction
- BiLSTM temporal learning
- Attention mechanism
- Streamlit web application
- Cross-dataset evaluation

## Dataset
### FaceForensics++
- 1000 Real Videos
- 1000 Fake Videos
- Total: 2000 Videos

### FakeAVCeleb
Used for cross-dataset evaluation across different races and genders.

## Repository Structure
Deepfake-detection/
├── 1221102371_DeepfakeDetectionPipeline.ipynb  # Full pipeline notebook (data download → preprocessing → training → evaluation)
├── streamlit_app2.py                            # Streamlit web app
├── requirements.txt                             # Python dependencies
└── README.md

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Leticia0403/Deepfake-detection.git
cd Deepfake-detection

# Install dependencies
pip install -r requirements.txt
```

---

## Run the Streamlit Web App

```bash
streamlit run streamlit_app2.py
```

**How to use:**
1. Upload a video file (MP4 format)
2. Click **Run Prediction**
3. The app will extract 10 frames, detect faces using MTCNN, and return a **Real / Fake** verdict with probability scores
4. The attention panel shows which frame the model focused on most

> Note: The app requires a face to be visible in the video. If no face is detected, an error message will appear.

---

## Model Download
Due to GitHub's file size limit, the trained model `.h5` file is hosted on Google Drive.

**Download:** https://drive.google.com/file/d/1T2oLUzevAf_QgReqxNzL_fiY8L3tEG3u/view?usp=sharing

After downloading, place the file in the same directory as `streamlit_app2.py` and update the model path in the app if needed.

---

## Requirements
Key dependencies (see `requirements.txt` for full list):
- Python 3.8+
- TensorFlow / Keras
- OpenCV
- MTCNN
- Streamlit
- NumPy, Pandas, Matplotlib, Scikit-learn

---

## Acknowledgements
- Dataset: FaceForensics++ (Rössler et al., 2019)
- Cross-dataset: FakeAVCeleb
