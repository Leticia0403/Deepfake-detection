# Deepfake Video Detection

## Project Overview
This project develops a deepfake video detection system using ResNet50, BiLSTM, and Attention Mechanism. The model is trained on the FaceForensics++ dataset and evaluated using both within-dataset testing and cross-dataset testing on FakeAVCeleb.

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

## Model Download 
Due to GitHub's file size restrictions, the trained model file (.h5) is hosted externally.
https://drive.google.com/file/d/1T2oLUzevAf_QgReqxNzL_fiY8L3tEG3u/view?usp=sharing
