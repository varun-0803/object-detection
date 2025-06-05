import cv2
import torch

# Load the YOLOv5 model (pretrained on COCO)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Video input and output
video_path = r"C:\Users\KML\Downloads\SampleVideo_1280x720_10mb.mp4"
output_path = r"C:\Users\KML\Downloads\output_detected.mp4"

# Open video capture
cap = cv2.VideoCapture(video_path)

# Get video info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]

    # Show the frame with detections
    cv2.imshow('YOLOv5 Detection', annotated_frame)

    # Write frame to output video
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 2 == 0:  # process every 2nd frame only
        # your frame processing and displaying here
        cv2.imshow('Video', frame)
    
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
