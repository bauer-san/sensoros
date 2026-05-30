# live_homography_overlay.py
import cv2
import numpy as np
import json

def load_calibration(cal_file="calibration.json"):
    with open(cal_file) as f:
        data = json.load(f)
    return np.array(data["homography_matrix"], dtype=np.float32)

def draw_world_grid(frame, H_inv, grid_spacing=1.0,
                    x_range=(0, 20), y_range=(0, 15)):
    """
    Project a metric grid from world space into the image.
    Gives you immediate visual confirmation that the homography is correct —
    grid lines should align with real-world features.
    """
    # Draw grid lines
    for x in np.arange(x_range[0], x_range[1] + grid_spacing, grid_spacing):
        pts_world = np.array(
            [[[x, y] for y in np.linspace(y_range[0], y_range[1], 50)]],
            dtype=np.float32
        )
        pts_img = cv2.perspectiveTransform(pts_world, H_inv)
        pts_img = pts_img[0].astype(int)

        for i in range(len(pts_img) - 1):
            p1, p2 = tuple(pts_img[i]), tuple(pts_img[i+1])
            # Only draw if points are within frame
            h, w = frame.shape[:2]
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(frame, p1, p2, (0, 255, 0), 1)

    for y in np.arange(y_range[0], y_range[1] + grid_spacing, grid_spacing):
        pts_world = np.array(
            [[[x, y] for x in np.linspace(x_range[0], x_range[1], 50)]],
            dtype=np.float32
        )
        pts_img = cv2.perspectiveTransform(pts_world, H_inv)
        pts_img = pts_img[0].astype(int)

        for i in range(len(pts_img) - 1):
            p1, p2 = tuple(pts_img[i]), tuple(pts_img[i+1])
            h, w = frame.shape[:2]
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(frame, p1, p2, (0, 255, 0), 1)

    # Label grid intersections
    for x in np.arange(x_range[0], x_range[1] + grid_spacing, grid_spacing):
        for y in np.arange(y_range[0], y_range[1] + grid_spacing, grid_spacing):
            pt_world = np.array([[[x, y]]], dtype=np.float32)
            pt_img = cv2.perspectiveTransform(pt_world, H_inv)[0][0].astype(int)
            h, w = frame.shape[:2]
            if 0 <= pt_img[0] < w and 0 <= pt_img[1] < h:
                cv2.circle(frame, tuple(pt_img), 3, (0, 200, 0), -1)
                if x % 2 == 0 and y % 2 == 0:  # label every 2m
                    cv2.putText(
                        frame, f"{x:.0f},{y:.0f}",
                        (pt_img[0] + 4, pt_img[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 200, 0), 1
                    )

mouse_world_pos = None

def mouse_callback(event, x, y, flags, param):
    global mouse_world_pos
    H = param
    if event == cv2.EVENT_MOUSEMOVE:
        pt = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, H)[0][0]
        mouse_world_pos = (float(world[0]), float(world[1]))

def main():
    H = load_calibration()
    H_inv = np.linalg.inv(H)  # world → image (for grid projection)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("Homography Validation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Homography Validation", mouse_callback, H)

    print("Move mouse over image to see ground plane coordinates")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw metric grid overlay
        draw_world_grid(frame, H_inv)

        # Show mouse position in world coordinates
        if mouse_world_pos:
            x_w, y_w = mouse_world_pos
            cv2.putText(
                frame,
                f"Ground plane: ({x_w:.2f}m, {y_w:.2f}m)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 0), 2
            )

        cv2.imshow("Homography Validation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()