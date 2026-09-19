import os
import tempfile

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# -------------------------------------------------
# Load YOLO Model
# -------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")


# -------------------------------------------------
# Image Detection
# -------------------------------------------------

def detect_image(image, model, original_filename="detected_image"):
    results = model(image, conf=0.25)

    annotated_image = results[0].plot()

    # Convert BGR -> RGB for Streamlit display
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Output filename
    output_path = os.path.join(
        "output",
        f"{original_filename}_detected.jpg"
    )

    # Save image
    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            annotated_image,
            cv2.COLOR_RGB2BGR
        )
    )

    return annotated_image, results[0], output_path


# -------------------------------------------------
# Video Detection
# -------------------------------------------------

def detect_video(video_path, model):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Output filename
    base_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    output_path = os.path.join(
        "output",
        f"{base_name}_detected.mp4"
    )

    # Video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=0.25)

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    return output_path


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.title("Personal Protective Equipment (PPE) Detection")

st.sidebar.header("Choose Input Type")

input_type = st.sidebar.radio(
    "Select input type",
    ["Image", "Video"]
)


# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = load_model()


# =================================================
# IMAGE MODE
# =================================================

if input_type == "Image":

    uploaded_image = st.sidebar.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        # Read image
        image = Image.open(uploaded_image)

        # Original image
        st.subheader("Original Image")

        st.image(
            image,
            width="stretch"
        )

        # Detection
        st.subheader("Detection Results")

        filename = os.path.splitext(
            uploaded_image.name
        )[0]

        annotated_image, results, image_output_path = detect_image(
            image,
            model,
            original_filename=filename
        )

        # Detected image
        st.image(
            annotated_image,
            caption="Detected Image",
            width="stretch"
        )

        # -------------------------------------------------
        # Detection Table
        # -------------------------------------------------

        st.subheader("Detected PPE Objects")

        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:

            class_ids = (
                boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                boxes.conf
                .cpu()
                .numpy()
            )

            detection_data = []

            for class_id, confidence in zip(
                class_ids,
                confidences
            ):

                class_name = model.names[class_id]

                detection_data.append({
                    "Detected Object": class_name,
                    "Confidence": f"{confidence * 100:.2f}%"
                })

            st.dataframe(
                detection_data,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No PPE objects detected in this image."
            )

        # -------------------------------------------------
        # Download
        # -------------------------------------------------

        with open(image_output_path, "rb") as file:

            st.download_button(
                label="Download Detected Image",
                data=file,
                file_name=os.path.basename(
                    image_output_path
                ),
                mime="image/jpeg"
            )


# =================================================
# VIDEO MODE
# =================================================

elif input_type == "Video":

    uploaded_video = st.sidebar.file_uploader(
        "Upload a Video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video:

        # Show original video
        st.subheader("Original Video")

        st.video(uploaded_video)

        # Process video
        with st.spinner(
            "Running object detection on video..."
        ):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(
                    uploaded_video.name
                )[1]
            ) as temp_input:

                temp_input.write(
                    uploaded_video.read()
                )

                video_path = temp_input.name

            output_video_path = detect_video(
                video_path,
                model
            )

        # Delete temporary input file
        try:
            os.remove(video_path)
        except OSError:
            pass

        # Success message
        st.success(
            "Detection completed successfully!"
        )

        # Processed video
        st.subheader("Detected Video")

        st.video(output_video_path)

        # Download
        with open(output_video_path, "rb") as file:

            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name=os.path.basename(
                    output_video_path
                ),
                mime="video/mp4"
            )