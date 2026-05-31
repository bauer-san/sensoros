import cv2
import os

url = os.environ.get("CAMERA_RTSP_URL", "")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Discard first few frames to get past buffer
for _ in range(5):
    cap.read()

ret, frame = cap.read()
if ret:
    cv2.imwrite("calibration/calibration_frame.jpg", frame)
    print(f"Saved calibration_frame.jpg — {frame.shape[1]}x{frame.shape[0]}")
else:
    print("Failed to capture frame")
cap.release()