import os
os.environ["STREAMLIT_WATCHER_IGNORE_FILES"] = ".*"

import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import time


def process_video(input_file, progress_callback, status_callback):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(input_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    output_path = temp_output.name
    temp_output.close()
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    frame_num = 0
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        frame_num += 1

        elapsed = time.time() - start_time
        fps_live = frame_num / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_num) / fps_live if fps_live > 0 else 0

        progress_callback(frame_num / total_frames)
        status_callback(f"Processed {frame_num}/{total_frames} frames | FPS: {fps_live:.2f} | ETA: {eta:.1f} seconds")

    cap.release()
    out.release()
    return output_path


def main():
    st.title("YOLOv8 Video Detection App")
    st.markdown("Upload a video, run object detection, and download the processed result.")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        st.markdown("Original video preview:")
        st.video(tfile.name)

        if st.button("Run Detection"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            processed_path = process_video(
                tfile.name,
                progress_callback=lambda p: progress_bar.progress(min(int(p * 100), 100)),
                status_callback=lambda txt: status_text.info(txt)
            )

            st.success("Processing complete.")
            st.markdown("Processed video preview:")
            st.video(processed_path)

            with open(processed_path, 'rb') as f:
                st.download_button("Download Processed Video", f, file_name="processed_output.avi")

            # Clean up temp files
            os.remove(tfile.name)
            os.remove(processed_path)

if __name__ == "__main__":
    main()
