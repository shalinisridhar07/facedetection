import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionLab - IVA Image Analysis",
    page_icon="🔍",
    layout="wide"
)


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
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is None:
    st.info("Please upload an image to begin analysis.")
    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(uploaded_file).convert("RGB")
image_array = np.array(image)

st.success("Image uploaded successfully!")


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.header("📊 Image Information")

width, height = image.size
channels = image_array.shape[2]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Width", f"{width} px")

with col2:
    st.metric("Height", f"{height} px")

with col3:
    st.metric("Channels", channels)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

st.header("📷 Original Image")

st.image(
    image,
    caption="Uploaded Image",
    width="stretch"
)


# ============================================================
# OPENCV CHECK
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
# 1️⃣ VIOLA-JONES FACE DETECTION
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
            (255, 0, 0),
            3
        )

        cv2.putText(
            viola_image,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    st.image(
        viola_image,
        caption="Viola-Jones Detection Result",
        width="stretch"
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

        st.warning(
            "No faces detected."
        )

except Exception as e:

    st.error(
        "Viola-Jones detection failed."
    )

    st.code(str(e))

    faces = []


# ============================================================
# 2️⃣ FACENET FACE EMBEDDING
# ============================================================

st.header("2️⃣ FaceNet Face Embedding")

try:

    from keras_facenet import FaceNet

    @st.cache_resource
    def load_facenet():

        model = FaceNet()

        return model


    with st.spinner("Loading FaceNet model..."):

        facenet = load_facenet()


    st.success(
        "FaceNet model loaded successfully."
    )


    if len(faces) == 0:

        st.warning(
            "No face detected. FaceNet embedding cannot be generated."
        )

    else:

        st.write(
            f"Generating embeddings for {len(faces)} detected face(s)..."
        )

        for i, (x, y, w, h) in enumerate(faces):

            face = image_array[
                y:y + h,
                x:x + w
            ]

            face_rgb = np.array(
                Image.fromarray(face).resize(
                    (160, 160)
                )
            )

            embedding = facenet.embeddings(
                [face_rgb]
            )[0]

            st.subheader(
                f"Face {i + 1} Embedding"
            )

            st.write(
                f"Embedding Dimension: **{len(embedding)}**"
            )

            st.write(
                "First 10 embedding values:"
            )

            st.code(
                str(embedding[:10])
            )

except Exception as e:

    st.error(
        "FaceNet could not be loaded."
    )

    st.code(
        str(e)
    )

    st.info(
        "Make sure keras-facenet and its dependencies "
        "are installed."
    )


# ============================================================
# 3️⃣ TEMPLATE MATCHING
# ============================================================

st.header("3️⃣ Template Matching")

template_file = st.file_uploader(
    "🧩 Upload a template image",
    type=["jpg", "jpeg", "png"],
    key="template"
)


if template_file is None:

    st.info(
        "Upload a template image to perform template matching."
    )

else:

    try:

        template_image = Image.open(
            template_file
        ).convert("RGB")

        template_array = np.array(
            template_image
        )

        st.subheader("Template Image")

        st.image(
            template_array,
            caption="Template",
            width="content"
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

        scales = np.linspace(
            0.5,
            1.5,
            21
        )

        for scale in scales:

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

            matching_image = image_array.copy()

            cv2.rectangle(
                matching_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                4
            )

            cv2.putText(
                matching_image,
                f"Match: {best_score:.2f}",
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            st.subheader(
                "Template Matching Result"
            )

            st.image(
                matching_image,
                caption="Best Template Match",
                width="stretch"
            )

            st.metric(
                "Matching Score",
                f"{best_score:.4f}"
            )

            if best_score >= 0.8:

                st.success(
                    "Strong template match found."
                )

            elif best_score >= 0.5:

                st.warning(
                    "Moderate template match found."
                )

            else:

                st.error(
                    "Weak template match."
                )

        else:

            st.warning(
                "Template could not be matched."
            )

    except Exception as e:

        st.error(
            "Template matching failed."
        )

        st.code(
            str(e)
        )


# ============================================================
# 4️⃣ DEEPFACE EMOTION ANALYSIS
# ============================================================

st.header("4️⃣ DeepFace Emotion Analysis")

try:

    from deepface import DeepFace

    st.success(
        "DeepFace package loaded successfully."
    )


    # Create temporary image file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name


    with st.spinner(
        "Analyzing facial emotion..."
    ):

        analysis = DeepFace.analyze(
            img_path=temp_path,
            actions=["emotion"],
            enforce_detection=False
        )


    # DeepFace may return either a list or dictionary
    if isinstance(
        analysis,
        list
    ):

        analysis = analysis[0]


    dominant_emotion = analysis.get(
        "dominant_emotion",
        "Unknown"
    )

    emotion_scores = analysis.get(
        "emotion",
        {}
    )


    # --------------------------------------------------------
    # Dominant Emotion
    # --------------------------------------------------------

    st.subheader(
        "😊 Dominant Emotion"
    )

    st.success(
        f"Detected Emotion: **{dominant_emotion.capitalize()}**"
    )


    # --------------------------------------------------------
    # Emotion Scores
    # --------------------------------------------------------

    st.subheader(
        "📊 Emotion Scores"
    )

    if emotion_scores:

        for emotion, score in emotion_scores.items():

            st.write(
                f"**{emotion.capitalize()}**: "
                f"{score:.2f}%"
            )

            st.progress(
                min(
                    max(
                        int(score),
                        0
                    ),
                    100
                )
            )


except Exception as e:

    st.error(
        "DeepFace emotion analysis failed."
    )

    st.code(
        str(e)
    )


# ============================================================
# ANALYSIS SUMMARY
# ============================================================

st.header("📋 Analysis Summary")

# Viola-Jones
if len(faces) > 0:

    st.write(
        f"**Viola-Jones:** "
        f"{len(faces)} face(s) detected"
    )

else:

    st.write(
        "**Viola-Jones:** No faces detected"
    )


# FaceNet
try:

    if len(faces) > 0:

        st.write(
            "**FaceNet:** Face embeddings generated"
        )

    else:

        st.write(
            "**FaceNet:** No face available for embedding"
        )

except Exception:

    st.write(
        "**FaceNet:** Unavailable"
    )


# Template Matching
if template_file is not None:

    st.write(
        "**Template Matching:** Tested"
    )

else:

    st.write(
        "**Template Matching:** Not tested"
    )


# DeepFace
st.write(
    "**DeepFace:** Emotion analysis performed"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    ---
    **VisionLab | Image and Video Analytics**

    Viola-Jones • FaceNet • Template Matching • DeepFace Emotion Analysis
    """
)