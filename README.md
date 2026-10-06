# 🔍 VisionLab – Intelligent Image Analysis System

VisionLab is a Streamlit-based computer vision web application that performs multiple image analysis techniques on an uploaded image.

## 🚀 Features

### 1. Viola-Jones Face Detection

Detects human faces using the classical Viola-Jones Haar Cascade algorithm.

### 2. FaceNet

Generates numerical face embeddings using the FaceNet deep-learning model.

### 3. Multi-Scale Template Matching

Searches for a smaller template image inside the uploaded image and displays the best matching location and similarity score.

### 4. DeepFace

Performs facial attribute analysis including:

- Estimated age
- Gender
- Dominant emotion
- Emotion confidence scores

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- TensorFlow
- Keras
- FaceNet
- DeepFace

## 📂 Project Structure

```text
IVA Assignment/
│
├── app.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run

Create and activate the virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

The application will open in the browser.

## 🌐 Live Demo

🚀 **[Try VisionLab Live](https://visionlab-image-analysis.streamlit.app)**

## 🖼️ How to Use

1. Upload a main image.
2. View Viola-Jones face detection results.
3. Generate FaceNet face embeddings.
4. Upload a template image for template matching.
5. View the best matching location and similarity score.
6. View DeepFace facial analysis results.

## 🎯 Project Objective

The objective of VisionLab is to demonstrate how classical computer vision techniques and modern deep-learning-based facial analysis can be integrated into a single interactive web application.

## 👩‍💻 Project

**IVA – Image and Video Analytics**

Developed using Python and Streamlit.
