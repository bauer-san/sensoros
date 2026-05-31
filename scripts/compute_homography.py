import cv2
import numpy as np
import json

def main():
    with open("calibration/image_points.json") as f:
        image_pts = np.array(json.load(f)["image_points"], dtype=np.float32)

    with open("calibration/world_points.json") as f:
        world_data = json.load(f)
        world_pts  = np.array(world_data["world_points"], dtype=np.float32)
        labels     = world_data["labels"]

    assert len(image_pts) == len(world_pts), \
        f"Point count mismatch: {len(image_pts)} image vs {len(world_pts)} world"

    H, mask = cv2.findHomography(
        image_pts, world_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=0.5
    )

    inliers = int(mask.sum())
    print(f"\nHomography computed from {inliers}/{len(image_pts)} inlier points")

    if inliers < 4:
        print("ERROR: Too few inliers — check point measurements")
        return

    # Reprojection errors
    print(f"\n{'Point':<22} {'World (m)':<22} {'Projected (m)':<22} {'Error (m)':<10} {'Status'}")
    print("-" * 85)

    errors = []
    for i, (img_pt, world_pt) in enumerate(zip(image_pts, world_pts)):
        pt        = np.array([[img_pt]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, H)[0][0]
        error     = float(np.linalg.norm(projected - world_pt))
        is_inlier = bool(mask[i])
        errors.append(error)

        status = "✅ inlier" if is_inlier else "❌ outlier"
        print(
            f"P{i+1:<2} {labels[i]:<18} "
            f"({world_pt[0]:+.2f}, {world_pt[1]:.2f}){'':<6} "
            f"({projected[0]:+.2f}, {projected[1]:.2f}){'':<6} "
            f"{error:.3f}m{'':<4} "
            f"{status}"
        )

    inlier_errors = [e for e, m in zip(errors, mask) if m]
    print(f"\nMean error (inliers): {np.mean(inlier_errors):.3f}m")
    print(f"Max error  (inliers): {np.max(inlier_errors):.3f}m")
    print(f"Std dev    (inliers): {np.std(inlier_errors):.3f}m")

    if np.mean(inlier_errors) < 0.15:
        print("✅ Excellent calibration")
    elif np.mean(inlier_errors) < 0.40:
        print("✅ Good — adequate for zone assignment")
    elif np.mean(inlier_errors) < 1.0:
        print("⚠️  Marginal — consider re-measuring outlier points")
    else:
        print("❌ Poor — recheck measurements and clicks")

    # Save
    output = {
        "homography_matrix": H.tolist(),
        "inliers": inliers,
        "total_points": len(image_pts),
        "mean_error_m": round(float(np.mean(inlier_errors)), 4),
        "metadata": {
            "origin":      world_data["notes"]["origin"],
            "x_axis":      world_data["notes"]["x_axis"],
            "y_axis":      world_data["notes"]["y_axis"],
            "units":       "meters",
            "image_resolution": [640, 480]
        }
    }

    with open("calibration/calibration.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to calibration/calibration.json")

if __name__ == "__main__":
    main()