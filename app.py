import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IVA Image Analysis",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔍 VisionLab")
st.subheader("Intelligent Image Analysis System")

st.write(
    "Upload an image and analyze it using "
    "Viola-Jones, FaceNet, Template Matching, "
    "and DeepFace Emotion Analysis."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# CACHED FACENET MODEL
# ============================================================

@st.cache_resource
def load_facenet():
    from keras_facenet import FaceNet
    return FaceNet()


# ============================================================
# CACHED DEEPFACE EMOTION MODEL
# ============================================================

@st.cache_resource
def load_deepface():
    from deepface import DeepFace

    # Build the Emotion model once.
    # DeepFace internally caches this model as well.
    DeepFace.build_model(
        "Emotion",
        task="facial_attribute"
    )

    return DeepFace


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image = Image.open(uploaded_file)

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    # --------------------------------------------------------
    # Convert PIL image to OpenCV format
    # --------------------------------------------------------

    image_rgb = np.array(image)

    # Handle RGBA images safely
    if image_rgb.shape[-1] == 4:

        image_rgb = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGBA2RGB
        )

    image_cv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    height, width, channels = image_cv.shape

    st.write(
        f"**Image Size:** {width} × {height}"
    )

    st.write(
        f"**Channels:** {channels}"
    )


    # ========================================================
    # 1️⃣ VIOLA-JONES FACE DETECTION
    # ========================================================

    st.header("1️⃣ Viola-Jones Face Detection")

    try:

        gray_image = cv2.cvtColor(
            image_cv,
            cv2.COLOR_BGR2GRAY
        )

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        face_cascade = cv2.CascadeClassifier(
            cascade_path
        )

        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(50, 50)
        )

        face_count = len(faces)

        # Draw rectangles
        viola_image = image_cv.copy()

        for (x, y, w, h) in faces:

            cv2.rectangle(
                viola_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        viola_image_rgb = cv2.cvtColor(
            viola_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            viola_image_rgb,
            caption="Viola-Jones Face Detection",
            width="stretch"
        )

        st.success(
            f"Viola-Jones detected {face_count} face(s)."
        )

    except Exception as e:

        face_count = 0

        st.error(
            f"Viola-Jones error: {str(e)}"
        )


    # ========================================================
    # 2️⃣ FACENET
    # ========================================================

    st.header("2️⃣ FaceNet Face Embeddings")

    try:

        embedder = load_facenet()

        # FaceNet expects RGB image
        facenet_image = image_rgb

        with st.spinner(
            "Generating FaceNet embeddings..."
        ):

            if face_count > 0:

                embeddings = []

                for (x, y, w, h) in faces:

                    face_crop = facenet_image[
                        y:y + h,
                        x:x + w
                    ]

                    if face_crop.size == 0:
                        continue

                    face_crop = cv2.resize(
                        face_crop,
                        (160, 160)
                    )

                    embedding = embedder.embeddings(
                        np.expand_dims(
                            face_crop,
                            axis=0
                        )
                    )

                    embeddings.append(
                        embedding[0]
                    )

                if embeddings:

                    st.success(
                        "Face embeddings generated"
                    )

                    st.write(
                        f"**Faces processed:** "
                        f"{len(embeddings)}"
                    )

                    st.write(
                        f"**Embedding size:** "
                        f"{len(embeddings[0])}"
                    )

                else:

                    st.warning(
                        "No valid face crop was available "
                        "for FaceNet."
                    )

            else:

                st.info(
                    "No face detected. "
                    "FaceNet embedding was not generated."
                )

    except Exception as e:

        st.error(
            f"FaceNet error: {str(e)}"
        )


    # ========================================================
    # 3️⃣ TEMPLATE MATCHING
    # ========================================================

    st.header("3️⃣ Template Matching")

    template_file = st.file_uploader(
        "📌 Upload a template image "
        "(optional)",
        type=["jpg", "jpeg", "png"],
        key="template"
    )

    if template_file is not None:

        try:

            template_pil = Image.open(
                template_file
            )

            template_rgb = np.array(
                template_pil
            )

            if template_rgb.shape[-1] == 4:

                template_rgb = cv2.cvtColor(
                    template_rgb,
                    cv2.COLOR_RGBA2RGB
                )

            template_cv = cv2.cvtColor(
                template_rgb,
                cv2.COLOR_RGB2BGR
            )

            main_gray = cv2.cvtColor(
                image_cv,
                cv2.COLOR_BGR2GRAY
            )

            template_gray = cv2.cvtColor(
                template_cv,
                cv2.COLOR_BGR2GRAY
            )

            template_height, template_width = (
                template_gray.shape
            )

            image_height, image_width = (
                main_gray.shape
            )

            if (
                template_height <= image_height
                and
                template_width <= image_width
            ):

                result = cv2.matchTemplate(
                    main_gray,
                    template_gray,
                    cv2.TM_CCOEFF_NORMED
                )

                min_val, max_val, min_loc, max_loc = (
                    cv2.minMaxLoc(result)
                )

                top_left = max_loc

                bottom_right = (
                    top_left[0] + template_width,
                    top_left[1] + template_height
                )

                matching_image = image_cv.copy()

                cv2.rectangle(
                    matching_image,
                    top_left,
                    bottom_right,
                    (255, 0, 0),
                    2
                )

                matching_image_rgb = cv2.cvtColor(
                    matching_image,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    matching_image_rgb,
                    caption="Template Matching Result",
                    width="stretch"
                )

                st.success(
                    "Template Matching: Tested"
                )

                st.write(
                    f"**Matching Score:** "
                    f"{max_val:.2f}"
                )

            else:

                st.warning(
                    "Template image must be smaller "
                    "than the uploaded image."
                )

        except Exception as e:

            st.error(
                f"Template Matching error: {str(e)}"
            )

    else:

        st.info(
            "Upload a template image to perform "
            "template matching."
        )


    # ========================================================
    # 4️⃣ DEEPFACE EMOTION ANALYSIS
    # ========================================================

    st.header("4️⃣ DeepFace Emotion Analysis")

    try:

        # Load the cached DeepFace + Emotion model
        DeepFace = load_deepface()

        st.success(
            "DeepFace package loaded successfully."
        )

        # ----------------------------------------------------
        # Convert uploaded image to OpenCV format
        # ----------------------------------------------------

        image_bytes = uploaded_file.getvalue()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        deepface_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if deepface_image is None:

            st.error(
                "Unable to read the uploaded image "
                "for DeepFace."
            )

        else:

            with st.spinner(
                "Analyzing facial emotion..."
            ):

                analysis = DeepFace.analyze(
                    img_path=deepface_image,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
                )

            # DeepFace may return a list
            # or a dictionary

            if isinstance(
                analysis,
                list
            ):

                if len(analysis) > 0:
                    analysis = analysis[0]
                else:
                    analysis = {}

            dominant_emotion = analysis.get(
                "dominant_emotion",
                "Unknown"
            )

            emotion_scores = analysis.get(
                "emotion",
                {}
            )

            # ------------------------------------------------
            # Dominant Emotion
            # ------------------------------------------------

            st.subheader(
                "😊 Dominant Emotion"
            )

            st.success(
                f"Detected Emotion: "
                f"**{dominant_emotion.capitalize()}**"
            )

            # ------------------------------------------------
            # Emotion Scores
            # ------------------------------------------------

            st.subheader(
                "📊 Emotion Scores"
            )

            if emotion_scores:

                for emotion, score in (
                    emotion_scores.items()
                ):

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

            else:

                st.info(
                    "No emotion scores were returned."
                )

    except Exception as e:

        st.error(
            "DeepFace emotion analysis "
            "could not be completed."
        )

        st.warning(
            f"Reason: {str(e)}"
        )


    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================

    st.header("📋 Analysis Summary")

    st.write(
        f"**Viola-Jones:** "
        f"{face_count} face(s) detected"
    )

    st.write(
        "**FaceNet:** Face embeddings generated"
    )

    st.write(
        "**Template Matching:** Tested"
    )

    st.write(
        "**DeepFace:** Emotion analysis performed"
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    st.markdown(
        "<div style='text-align:center;'>"
        "<b>VisionLab | Image and Video Analytics</b>"
        "<br>"
        "Viola-Jones • FaceNet • Template Matching • "
        "DeepFace Emotion Analysis"
        "</div>",
        unsafe_allow_html=True
    )
