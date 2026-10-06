# 🔍 VisionLab – Image and Face Analysis

VisionLab is a **Streamlit-based Image and Video Analytics (IVA) application** that performs multiple image and face analysis techniques on an uploaded image.

The application combines traditional computer vision methods with deep learning-based face analysis to demonstrate different approaches used in image processing and facial analysis.

## 🚀 Live Demo

**[Open VisionLab – Live Demo](https://facedetection-6zfappzse2bvekmickgrqrk.streamlit.app/)**

---

## ✨ Features

### 1. 🎯 Template Matching

Template Matching is used to locate a smaller template image inside a larger input image.

- Upload the main image.
- Upload a template image.
- Compare the template with different regions of the input image.
- Identify the best matching location.
- Display the matching region using a bounding box.
- Display the matching score.

**Technology:** OpenCV

### 2. 🧑 FaceNet

FaceNet is a deep learning-based face recognition technique that converts a detected face into a numerical **face embedding**.

- Detect faces from the uploaded image.
- Resize the detected face.
- Generate a numerical embedding using FaceNet.
- Display the face embedding information.

**Technology:** Keras-FaceNet

### 3. 😊 DeepFace

DeepFace is a deep learning-based facial analysis framework.

In this application, DeepFace is used for **facial emotion analysis**.

- Analyze the uploaded image.
- Detect facial expressions.
- Predict different emotions.
- Display the dominant emotion.
- Display emotion confidence scores.

**Technology:** DeepFace + TensorFlow/Keras

### 4. 👤 Viola–Jones Face Detection

The Viola–Jones algorithm is a classical object detection technique used for **face detection**.

The application uses OpenCV's pre-trained Haar Cascade classifier.

- Convert the image to grayscale.
- Detect faces using the Haar Cascade classifier.
- Draw bounding boxes around detected faces.
- Display the total number of detected faces.

**Technology:** OpenCV Haar Cascade

---

## 🔄 Application Workflow

```text
              Upload Image
                   │
                   ▼
        ┌─────────────────────┐
        │   VisionLab App     │
        └─────────────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
  Viola–Jones   FaceNet    DeepFace
 Face Detection Embedding  Emotion Analysis
       │           │           │
       └───────────┼───────────┘
                   │
                   ▼
          Analysis Results

          Template Matching
                   │
                   ▼
            Best Match Region
```

---

## 🛠️ Technologies Used

| Technology                | Purpose                              |
| ------------------------- | ------------------------------------ |
| Python                    | Programming language                 |
| Streamlit                 | Web application framework            |
| OpenCV                    | Image processing and computer vision |
| NumPy                     | Numerical and array operations       |
| Pillow                    | Image handling                       |
| Keras-FaceNet             | Face embedding generation            |
| DeepFace                  | Facial emotion analysis              |
| TensorFlow                | Deep learning backend                |
| Git                       | Version control                      |
| GitHub                    | Source code hosting                  |
| Streamlit Community Cloud | Application deployment               |

---

## 📁 Project Structure

```text
facedetection/
│
├── app.py
├── README.md
├── requirements.txt
├── packages.txt
└── .gitignore
```

---

## 💻 Run the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/shalinisridhar07/facedetection.git
```

### 2. Open the project directory

```bash
cd facedetection
```

### 3. Create a virtual environment

```bash
python -m venv visionenv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\visionenv\Scripts\Activate.ps1
```

### 5. Install the required packages

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 📊 Operations Summary

| Operation         | Purpose                           | Method        |
| ----------------- | --------------------------------- | ------------- |
| Template Matching | Locate a template inside an image | OpenCV        |
| FaceNet           | Generate face embeddings          | Deep Learning |
| DeepFace          | Analyze facial emotions           | Deep Learning |
| Viola–Jones       | Detect faces                      | Haar Cascade  |

---

## 🎓 Project Purpose

This project was developed as part of an **Image and Video Analytics (IVA)** practical/academic project to demonstrate different computer vision and facial analysis techniques using Python.

The application provides a single web interface where users can upload an image and perform multiple analysis operations.

---

## 👩‍💻 Author

**Shalini Sridhar**

GitHub:
https://github.com/shalinisridhar07

---

## 🔗 Links

**GitHub Repository:**
https://github.com/shalinisridhar07/facedetection

**Live Demo:**
https://facedetection-6zfappzse2bvekmickgrqrk.streamlit.app/
