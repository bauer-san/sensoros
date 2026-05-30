# capture_calibration_image.py
# Run this first to get a still frame for point selection

import cv2

# For USB camera (Logitech BRIO etc.)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# For RTSP IP camera
# cap = cv2.VideoCapture("rtsp://admin:password@192.168.1.64/stream1")

print("Press SPACE to capture, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Calibration Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        cv2.imwrite("calibration_frame.jpg", frame)
        print("Saved calibration_frame.jpg")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()