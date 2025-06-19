from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(r"C:\Users\KML\Downloads\SAMPLE (1).avi")
w = int(cap.get(3))
h = int(cap.get(4))
fps = cap.get(5)
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    result = results[0].plot()
    cv2.imshow("detection", result)
    out.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()                                                      