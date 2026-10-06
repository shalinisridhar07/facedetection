import streamlit as st
import cv2
import numpy as np
import tempfile
import os

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
# SIDEBAR
# ============================================================

st.sidebar.title("🔬 Analysis Methods")

st.sidebar.write("✅ Viola-Jones Face Detection")
st.sidebar.write("✅ FaceNet Face Embedding")
st.sidebar.write("✅ Multi-Scale Template Matching")
st.sidebar.write("✅ DeepFace Analysis")

st.sidebar.markdown("---")

st.sidebar.info(
    "IVA Project\n\n"
    "Computer Vision and Image Analysis"
)


# ============================================================
# LOAD VIOLA-JONES CLASSIFIER
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Could not load Viola-Jones Haar Cascade.")
    st.stop()


# ============================================================
# LOAD FACENET
# ============================================================

@st.cache_resource
def load_facenet():

    try:
        from keras_facenet import FaceNet

        model = FaceNet()

        return model

    except Exception as e:

        st.error("FaceNet could not be loaded.")

        st.code(str(e))

        return None


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload Main Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    pil_image = Image.open(uploaded_file).convert("RGB")

    image = np.array(pil_image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    st.success("Image uploaded successfully!")


    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    st.header("📊 Image Information")

    height, width, channels = image.shape

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Width",
            f"{width} px"
        )

    with col2:
        st.metric(
            "Height",
            f"{height} px"
        )

    with col3:
        st.metric(
            "Channels",
            channels
        )


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    st.header("📷 Original Image")

    st.image(
        pil_image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # ========================================================
    # 1. VIOLA-JONES
    # ========================================================

    st.header("1️⃣ Viola-Jones Face Detection")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(60, 60)
    )

    viola_image = image.copy()

    for i, (x, y, w, h) in enumerate(faces):

        cv2.rectangle(
            viola_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            viola_image,
            f"Face {i + 1}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    viola_image_rgb = cv2.cvtColor(
        viola_image,
        cv2.COLOR_BGR2RGB
    )


    col1, col2 = st.columns(2)

    with col1:

        st.image(
            viola_image_rgb,
            caption="Viola-Jones Detection Result",
            use_container_width=True
        )

    with col2:

        st.metric(
            "Faces Detected",
            len(faces)
        )

        if len(faces) == 0:

            st.warning(
                "No face detected."
            )

        elif len(faces) == 1:

            st.success(
                "1 face detected."
            )

        else:

            st.success(
                f"{len(faces)} faces detected."
            )


    # ========================================================
    # 2. FACENET
    # ========================================================

    st.header("2️⃣ FaceNet Face Embedding")

    facenet = load_facenet()

    embeddings = []

    if facenet is None:

        st.warning(
            "FaceNet is unavailable. "
            "Check the installation."
        )

    elif len(faces) == 0:

        st.warning(
            "No face detected. "
            "FaceNet cannot generate an embedding."
        )

    else:

        try:

            for i, (x, y, w, h) in enumerate(faces):

                face = image[
                    y:y + h,
                    x:x + w
                ]

                if face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                embedding = facenet.embeddings(
                    [face_rgb]
                )[0]

                embeddings.append(
                    embedding
                )


            st.success(
                f"FaceNet generated "
                f"{len(embeddings)} embedding(s)."
            )


            for i, embedding in enumerate(embeddings):

                with st.expander(
                    f"Face {i + 1} Embedding"
                ):

                    st.write(
                        "Embedding Dimension:"
                    )

                    st.write(
                        len(embedding)
                    )

                    st.write(
                        "First 10 Embedding Values:"
                    )

                    st.code(
                        str(embedding[:10])
                    )

        except Exception as e:

            st.error(
                "FaceNet processing failed."
            )

            st.code(
                str(e)
            )


    # ========================================================
    # 3. TEMPLATE MATCHING
    # ========================================================

    st.header("3️⃣ Template Matching")

    st.write(
        "Upload a smaller image cropped from the main image. "
        "The application searches for that template inside "
        "the uploaded main image."
    )

    template_file = st.file_uploader(
        "🖼️ Upload Template Image",
        type=["jpg", "jpeg", "png"],
        key="template"
    )


    if template_file is not None:

        template_pil = Image.open(
            template_file
        ).convert("RGB")

        template = np.array(
            template_pil
        )

        template = cv2.cvtColor(
            template,
            cv2.COLOR_RGB2BGR
        )


        gray_main = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray_template = cv2.cvtColor(
            template,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # MULTI-SCALE TEMPLATE MATCHING
        # ----------------------------------------------------

        best_score = -1

        best_location = None

        best_size = None


        # Check different template sizes
        for scale in np.arange(
            0.5,
            1.51,
            0.1
        ):

            new_width = int(
                gray_template.shape[1] * scale
            )

            new_height = int(
                gray_template.shape[0] * scale
            )


            if new_width <= 0 or new_height <= 0:

                continue


            if (
                new_width > gray_main.shape[1]
                or
                new_height > gray_main.shape[0]
            ):

                continue


            resized_template = cv2.resize(
                gray_template,
                (new_width, new_height)
            )


            result = cv2.matchTemplate(
                gray_main,
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


        # ----------------------------------------------------
        # DRAW BEST MATCH
        # ----------------------------------------------------

        if best_location is not None:

            top_left = best_location

            bottom_right = (
                top_left[0] + best_size[0],
                top_left[1] + best_size[1]
            )


            matching_image = image.copy()


            cv2.rectangle(
                matching_image,
                top_left,
                bottom_right,
                (255, 0, 0),
                3
            )


            cv2.putText(
                matching_image,
                f"Match: {best_score:.2f}",
                (
                    top_left[0],
                    max(top_left[1] - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )


            matching_image_rgb = cv2.cvtColor(
                matching_image,
                cv2.COLOR_BGR2RGB
            )


            st.image(
                matching_image_rgb,
                caption="Multi-Scale Template Matching Result",
                use_container_width=True
            )


            # ------------------------------------------------
            # RESULT INFORMATION
            # ------------------------------------------------

            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Matching Score",
                    f"{best_score:.2f}"
                )


            with col2:

                st.write(
                    "Best Match Location:"
                )

                st.code(
                    str(top_left)
                )


            # ------------------------------------------------
            # MATCH QUALITY
            # ------------------------------------------------

            if best_score >= 0.70:

                st.success(
                    "🟢 Strong template match found!"
                )

            elif best_score >= 0.50:

                st.warning(
                    "🟡 Moderate template similarity found."
                )

            else:

                st.info(
                    "🔵 Low template similarity. "
                    "For best results, use a template "
                    "cropped directly from the main image."
                )


    else:

        st.info(
            "Upload a template image to perform "
            "template matching."
        )


    # ========================================================
    # 4. DEEPFACE
    # ========================================================

    st.header("4️⃣ DeepFace Analysis")

    st.write(
        "DeepFace performs facial attribute analysis "
        "including age, gender and emotion."
    )


    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    temp_path = temp_file.name

    temp_file.close()


    cv2.imwrite(
        temp_path,
        image
    )


    try:

        from deepface import DeepFace


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


        if isinstance(
            analysis,
            list
        ):

            analysis = analysis[0]


        age = analysis.get(
            "age",
            "N/A"
        )

        gender = analysis.get(
            "dominant_gender",
            "N/A"
        )

        emotion = analysis.get(
            "dominant_emotion",
            "N/A"
        )


        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Estimated Age",
                age
            )


        with col2:

            st.metric(
                "Gender",
                gender
            )


        with col3:

            st.metric(
                "Dominant Emotion",
                emotion
            )


        # ----------------------------------------------------
        # EMOTION SCORES
        # ----------------------------------------------------

        emotion_scores = analysis.get(
            "emotion",
            {}
        )


        if emotion_scores:

            st.subheader(
                "😊 Emotion Scores"
            )

            st.bar_chart(
                emotion_scores
            )


    except Exception as e:

        st.error(
            "DeepFace analysis failed."
        )

        st.code(
            str(e)
        )


    finally:

        if os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.markdown("---")

    st.header("📋 Analysis Summary")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Viola-Jones",
            f"{len(faces)} Face(s)"
        )


    with col2:

        st.metric(
            "FaceNet",
            f"{len(embeddings)} Embedding(s)"
        )


    with col3:

        if template_file is not None and best_location is not None:

            st.metric(
                "Template Score",
                f"{best_score:.2f}"
            )

        else:

            st.metric(
                "Template Matching",
                "Not Tested"
            )


    with col4:

        st.metric(
            "DeepFace",
            "Analyzed"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "VisionLab | IVA Project | "
    "Viola-Jones + FaceNet + "
    "Multi-Scale Template Matching + DeepFace"
)
