# collect_image_points.py
import cv2
import numpy as np
import json

image_points = []
current_image = None

def mouse_callback(event, x, y, flags, param):
    global image_points, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        image_points.append([x, y])
        idx = len(image_points)

        # Draw marker on image
        display = current_image.copy()
        cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(
            display, f"P{idx} ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

        # Redraw all previous points
        for i, pt in enumerate(image_points):
            cv2.circle(display, tuple(pt), 8, (0, 255, 0), -1)
            cv2.putText(
                display, f"P{i+1}",
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )

        current_image = display
        print(f"Point P{idx}: pixel ({x}, {y})")

def main():
    global current_image

    img = cv2.imread("calibration_frame.jpg")
    current_image = img.copy()

    cv2.namedWindow("Select Calibration Points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Calibration Points", mouse_callback)

    print("Click on calibration points in order.")
    print("Record each point's real-world coordinates as you go.")
    print("Press S to save and quit, U to undo last point.")

    while True:
        cv2.imshow("Select Calibration Points", current_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            output = {"image_points": image_points}
            with open("image_points.json", "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved {len(image_points)} points to image_points.json")
            print("Now record their real-world coordinates in world_points.json")
            break

        elif key == ord('u') and image_points:
            image_points.pop()
            current_image = img.copy()
            # Redraw remaining points
            for i, pt in enumerate(image_points):
                cv2.circle(current_image, tuple(pt), 8, (0, 255, 0), -1)
                cv2.putText(
                    current_image, f"P{i+1}",
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )
            print(f"Undid last point. {len(image_points)} points remaining.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()