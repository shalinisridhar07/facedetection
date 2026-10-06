import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image

st.set_page_config(
    page_title="VisionLab - IVA Image Analysis",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# OPENCV
# ============================================================

try:
    import cv2
    CV2_AVAILABLE = True
    CV2_ERROR = None
except Exception as e:
    CV2_AVAILABLE = False
    CV2_ERROR = str(e)

if not CV2_AVAILABLE:
    st.title("🔍 VisionLab")
    st.error("OpenCV could not be loaded.")
    st.code(CV2_ERROR)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🔍 VisionLab")
st.subheader("Intelligent Image Analysis System")

st.write(
    "Upload an image and analyze it using four computer vision "
    "techniques: Viola-Jones, FaceNet, Template Matching, and DeepFace."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload an image to begin analysis.")
    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

image = Image.open(uploaded_file).convert("RGB")

image_array = np.array(image)

st.success("Image uploaded successfully!")


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.header("📊 Image Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Width", f"{image.width} px")

with col2:
    st.metric("Height", f"{image.height} px")

with col3:
    st.metric("Channels", "3")


# ============================================================
# ORIGINAL IMAGE
# ============================================================

st.header("📷 Original Image")

st.image(
    image,
    width="stretch",
    caption="Uploaded Image"
)


# ============================================================
# 1. VIOLA-JONES
# ============================================================

st.header("1️⃣ Viola-Jones Face Detection")

try:

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(60, 60)
    )

    viola_image = image_array.copy()

    for (x, y, w, h) in faces:

        cv2.rectangle(
            viola_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            viola_image,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    st.image(
        viola_image,
        width="stretch",
        caption="Viola-Jones Detection Result"
    )

    st.metric(
        "Faces Detected",
        len(faces)
    )

    if len(faces) > 0:
        st.success(
            f"{len(faces)} face(s) detected."
        )
    else:
        st.warning("No faces detected.")

except Exception as e:

    faces = []

    st.error("Viola-Jones failed.")
    st.code(str(e))


# ============================================================
# 2. FACENET
# ============================================================

st.header("2️⃣ FaceNet Face Embedding")

facenet = None

try:

    from keras_facenet import FaceNet

    @st.cache_resource
    def load_facenet():

        model = FaceNet()

        return model

    facenet = load_facenet()

    st.success("✅ FaceNet model loaded successfully.")

except Exception as e:

    st.error("❌ FaceNet could not be loaded.")

    st.code(str(e))

    st.info(
        "If this shows 'No module named keras_facenet', "
        "make sure keras-facenet is present in requirements.txt."
    )


# ============================================================
# FACENET PROCESSING
# ============================================================

if facenet is not None and len(faces) > 0:

    try:

        embeddings = []

        for i, (x, y, w, h) in enumerate(faces):

            face = image_array[
                y:y+h,
                x:x+w
            ]

            face_pil = Image.fromarray(face)

            face_rgb = np.array(
                face_pil.resize((160, 160))
            )

            embedding = facenet.embeddings(
                [face_rgb]
            )[0]

            embeddings.append(embedding)

            st.subheader(
                f"Face {i + 1} Embedding"
            )

            st.write(
                f"Embedding Dimension: "
                f"{len(embedding)}"
            )

            st.write(
                "First 10 embedding values:"
            )

            st.code(
                str(embedding[:10])
            )

        st.success(
            f"FaceNet generated embeddings for "
            f"{len(embeddings)} face(s)."
        )

    except Exception as e:

        st.error("FaceNet processing failed.")
        st.code(str(e))

elif facenet is not None:

    st.info(
        "FaceNet is ready, but no faces were detected "
        "by Viola-Jones."
    )


# ============================================================
# 3. TEMPLATE MATCHING
# ============================================================

st.header("3️⃣ Template Matching")

template_file = st.file_uploader(
    "📌 Upload a Template Image",
    type=["jpg", "jpeg", "png"],
    key="template"
)

template_score = None

if template_file is not None:

    try:

        template = Image.open(
            template_file
        ).convert("RGB")

        template_array = np.array(template)

        st.image(
            template,
            width="content",
            caption="Template Image"
        )

        main_gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        template_gray = cv2.cvtColor(
            template_array,
            cv2.COLOR_RGB2GRAY
        )

        best_score = -1
        best_location = None
        best_size = None

        for scale in np.arange(
            0.5,
            1.51,
            0.1
        ):

            new_width = int(
                template_gray.shape[1] * scale
            )

            new_height = int(
                template_gray.shape[0] * scale
            )

            if new_width < 10 or new_height < 10:
                continue

            if (
                new_width > main_gray.shape[1]
                or
                new_height > main_gray.shape[0]
            ):
                continue

            resized_template = cv2.resize(
                template_gray,
                (new_width, new_height)
            )

            result = cv2.matchTemplate(
                main_gray,
                resized_template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_val, _, max_loc = cv2.minMaxLoc(
                result
            )

            if max_val > best_score:

                best_score = max_val
                best_location = max_loc
                best_size = (
                    new_width,
                    new_height
                )

        if best_location is not None:

            x, y = best_location
            w, h = best_size

            result_image = image_array.copy()

            cv2.rectangle(
                result_image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                4
            )

            st.image(
                result_image,
                width="stretch",
                caption="Template Matching Result"
            )

            template_score = best_score

            st.metric(
                "Best Matching Score",
                f"{best_score:.4f}"
            )

            if best_score >= 0.70:

                st.success(
                    "Strong template match detected."
                )

            elif best_score >= 0.50:

                st.warning(
                    "Moderate template match detected."
                )

            else:

                st.info(
                    "Low template similarity."
                )

    except Exception as e:

        st.error(
            "Template Matching failed."
        )

        st.code(str(e))

else:

    st.info(
        "Upload a template image to perform template matching."
    )


# ============================================================
# 4. DEEPFACE
# ============================================================

st.header("4️⃣ DeepFace Analysis")

deepface_available = False
DeepFace = None

try:

    from deepface import DeepFace

    deepface_available = True

    st.success(
        "✅ DeepFace package loaded successfully."
    )

except Exception as e:

    st.error(
        "❌ DeepFace could not be loaded."
    )

    st.code(str(e))

    st.info(
        "Check the TensorFlow, Keras, tf-keras, "
        "and DeepFace versions in requirements.txt."
    )


# ============================================================
# DEEPFACE PROCESSING
# ============================================================

if deepface_available:

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            image.save(
                temp_file.name,
                format="JPEG"
            )

            temp_path = temp_file.name

        with st.spinner(
            "DeepFace is analyzing the image..."
        ):

            analysis = DeepFace.analyze(
                img_path=temp_path,
                actions=[
                    "age",
                    "gender",
                    "emotion"
                ],
                enforce_detection=False
            )

        # DeepFace can return either a dictionary
        # or a list of dictionaries.

        if isinstance(analysis, list):

            analysis_results = analysis

        else:

            analysis_results = [analysis]

        st.success(
            f"DeepFace analyzed "
            f"{len(analysis_results)} face/result(s)."
        )

        for i, result in enumerate(
            analysis_results
        ):

            st.subheader(
                f"Face {i + 1}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                age = result.get(
                    "age",
                    "N/A"
                )

                st.metric(
                    "Age",
                    str(age)
                )

            with col2:

                gender = result.get(
                    "dominant_gender",
                    "N/A"
                )

                st.metric(
                    "Gender",
                    str(gender)
                )

            with col3:

                emotion = result.get(
                    "dominant_emotion",
                    "N/A"
                )

                st.metric(
                    "Emotion",
                    str(emotion)
                )

            emotions = result.get(
                "emotion",
                {}
            )

            if emotions:

                st.subheader(
                    "Emotion Scores"
                )

                emotion_values = {
                    k: round(float(v), 2)
                    for k, v in emotions.items()
                }

                st.json(
                    emotion_values
                )

    except Exception as e:

        st.error(
            "DeepFace analysis failed."
        )

        st.code(str(e))

    finally:

        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# FINAL SUMMARY
# ============================================================

st.header("📋 Analysis Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:

    st.write(
        f"**Viola-Jones:** "
        f"{len(faces)} face(s) detected"
    )

    if facenet is not None:

        st.write(
            f"**FaceNet:** "
            f"Model loaded"
        )

    else:

        st.write(
            "**FaceNet:** Unavailable"
        )


with summary_col2:

    if template_score is not None:

        st.write(
            f"**Template Matching:** "
            f"{template_score:.4f}"
        )

    else:

        st.write(
            "**Template Matching:** "
            "Not tested"
        )

    if deepface_available:

        st.write(
            "**DeepFace:** Model loaded"
        )

    else:

        st.write(
            "**DeepFace:** Unavailable"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "VisionLab | Image and Video Analytics | "
    "Viola-Jones • FaceNet • Template Matching • DeepFace"
)
