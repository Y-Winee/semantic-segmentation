import cv2
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os

from model import load_model
from predict import predict
from visualization import colorize, overlay


# ============================================================
# LOAD MODEL ONCE
# ============================================================

@st.cache_resource
def get_model():
    return load_model()


model = get_model()


# ============================================================
# PAGE
# ============================================================

st.title("DeepLabV3 Semantic Segmentation")


input_type = st.radio(
    "Choose input",
    ["Image", "Video"],
    horizontal=True
)


# ============================================================
# IMAGE MODE
# ============================================================

if input_type == "Image":

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        original_h, original_w = image_np.shape[:2]

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        mask = predict(
            model,
            image_np
        )

        # ----------------------------------------------------
        # COLORIZE MASK
        # ----------------------------------------------------

        color_mask = colorize(mask)

        color_mask = cv2.resize(
            color_mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        overlay_img = overlay(
            image_np,
            color_mask
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original")

            st.image(
                image_np,
                channels="RGB",
                use_container_width=True
            )

        with col2:

            st.subheader("Segmentation")

            st.image(
                overlay_img,
                channels="RGB",
                use_container_width=True
            )


# ============================================================
# VIDEO MODE
# ============================================================

else:

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        # ----------------------------------------------------
        # SAVE INPUT VIDEO
        # ----------------------------------------------------

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(
            uploaded_video.getbuffer()
        )

        temp_input.close()

        input_video_path = temp_input.name

        # ----------------------------------------------------
        # OPEN VIDEO
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            input_video_path
        )

        if not cap.isOpened():

            st.error("Could not open video.")

        else:

            # ------------------------------------------------
            # VIDEO INFORMATION
            # ------------------------------------------------

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            duration = (
                total_frames / fps
                if fps > 0
                else 0
            )

            st.write(
                f"Video FPS: {fps:.2f}"
            )

            st.write(
                f"Resolution: {width} × {height}"
            )

            st.write(
                f"Total Frames: {total_frames}"
            )

            st.write(
                f"Duration: {duration:.1f} seconds"
            )

            # ------------------------------------------------
            # FRAME SKIPPING
            # ------------------------------------------------

            FRAME_SKIP = st.slider(
                "Process every Nth frame",
                min_value=1,
                max_value=6,
                value=3
            )

            # ------------------------------------------------
            # OUTPUT VIDEO
            # ------------------------------------------------

            temp_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_output.close()

            output_video_path = temp_output.name

            # ------------------------------------------------
            # VIDEO WRITER
            # ------------------------------------------------

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (width, height)
            )

            if not out.isOpened():

                st.error(
                    "Could not create output video."
                )

                cap.release()

            else:

                # ------------------------------------------------
                # DISPLAY
                # ------------------------------------------------

                frame_placeholder = st.empty()

                progress_bar = st.progress(0)

                status_text = st.empty()

                # ------------------------------------------------
                # PROCESSING
                # ------------------------------------------------

                frame_count = 0

                last_color_mask = None

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_count += 1

                    # --------------------------------------------
                    # BGR -> RGB
                    # --------------------------------------------

                    frame_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    original_h, original_w = (
                        frame_rgb.shape[:2]
                    )

                    # ============================================
                    # SEGMENTATION
                    # ============================================

                    if (
                        last_color_mask is None
                        or frame_count % FRAME_SKIP == 1
                    ):

                        mask = predict(
                            model,
                            frame_rgb
                        )

                        last_color_mask = colorize(
                            mask
                        )

                        last_color_mask = cv2.resize(
                            last_color_mask,
                            (
                                original_w,
                                original_h
                            ),
                            interpolation=cv2.INTER_NEAREST
                        )

                    # ============================================
                    # OVERLAY
                    # ============================================

                    overlay_img = overlay(
                        frame_rgb,
                        last_color_mask
                    )

                    # ============================================
                    # DISPLAY
                    # ============================================

                    frame_placeholder.image(
                        overlay_img,
                        channels="RGB",
                        use_container_width=True
                    )

                    # ============================================
                    # CONVERT RGB -> BGR FOR VIDEO
                    # ============================================

                    output_frame = cv2.cvtColor(
                        overlay_img,
                        cv2.COLOR_RGB2BGR
                    )

                    # ============================================
                    # WRITE FRAME
                    # ============================================

                    out.write(
                        output_frame
                    )

                    # ============================================
                    # PROGRESS
                    # ============================================

                    progress = (
                        frame_count / total_frames
                        if total_frames > 0
                        else 0
                    )

                    progress_bar.progress(
                        min(progress, 1.0)
                    )

                    status_text.write(
                        f"Processing frame "
                        f"{frame_count}/{total_frames} "
                        f"({progress * 100:.1f}%)"
                    )

                # ------------------------------------------------
                # CLEANUP
                # ------------------------------------------------

                cap.release()
                out.release()

                progress_bar.empty()
                status_text.empty()

                # ------------------------------------------------
                # COMPLETION
                # ------------------------------------------------

                st.success(
                    "Video segmentation complete."
                )

                # ------------------------------------------------
                # DOWNLOAD BUTTON
                # ------------------------------------------------

                with open(
                    output_video_path,
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()

                st.download_button(
                    label="Download Segmented Video",
                    data=video_bytes,
                    file_name="segmented_video.mp4",
                    mime="video/mp4"
                )

                # ------------------------------------------------
                # CLEAN TEMP FILES
                # ------------------------------------------------

                try:
                    os.remove(
                        input_video_path
                    )

                    os.remove(
                        output_video_path
                    )

                except OSError:
                    pass