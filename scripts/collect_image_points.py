import cv2
import numpy as np
import json
import os

# Load world points to show labels while clicking
with open("calibration/world_points.json") as f:
    world_data = json.load(f)

labels       = world_data["labels"]
image_points = []
current_img  = None
idx          = 0

def mouse_callback(event, x, y, flags, param):
    global image_points, current_img, idx

    if event == cv2.EVENT_LBUTTONDOWN:
        if idx >= len(labels):
            print("All points collected — press S to save")
            return

        image_points.append([x, y])
        print(f"  P{idx+1} {labels[idx]}: pixel ({x}, {y})")
        idx += 1

        # Redraw
        current_img = param.copy()
        for i, pt in enumerate(image_points):
            color = (0, 255, 0) if i < idx else (0, 200, 200)
            cv2.circle(current_img, tuple(pt), 8, color, -1)
            cv2.putText(
                current_img,
                f"P{i+1} {labels[i]}",
                (pt[0] + 10, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1
            )

        if idx < len(labels):
            print(f"  Next: click P{idx+1} — {labels[idx]}")
        else:
            print("All points collected — press S to save, U to undo")

def main():
    global current_img, idx, image_points

    img = cv2.imread("calibration/calibration_frame.jpg")
    if img is None:
        print("ERROR: calibration_frame.jpg not found")
        return

    current_img = img.copy()

    # Draw next-point prompt on image
    def draw_prompt(image):
        overlay = image.copy()
        if idx < len(labels):
            cv2.putText(
                overlay,
                f"Click: P{idx+1} — {labels[idx]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2
            )
        return overlay

    cv2.namedWindow("Calibration — Click Points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration — Click Points", 1280, 960)
    cv2.setMouseCallback(
        "Calibration — Click Points",
        mouse_callback,
        img
    )

    print(f"\nCalibration — {len(labels)} points to collect")
    print(f"Click each marker IN ORDER as listed:")
    for i, label in enumerate(labels):
        print(f"  P{i+1}: {label}")
    print(f"\nControls: S = save   U = undo last   Q = quit")
    print(f"\nFirst: click P1 — {labels[0]}\n")

    while True:
        display = draw_prompt(current_img)
        cv2.imshow("Calibration — Click Points", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s') and len(image_points) >= 4:
            output = {"image_points": image_points}
            with open("calibration/image_points.json", "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved {len(image_points)} points to "
                  f"calibration/image_points.json")
            break

        elif key == ord('u') and image_points:
            image_points.pop()
            idx -= 1
            current_img = img.copy()
            for i, pt in enumerate(image_points):
                cv2.circle(current_img, tuple(pt), 8, (0, 255, 0), -1)
                cv2.putText(
                    current_img,
                    f"P{i+1} {labels[i]}",
                    (pt[0] + 10, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1
                )
            print(f"Undid P{idx+1}. Click P{idx+1} — {labels[idx]} again")

        elif key == ord('q'):
            print("Quit without saving")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()