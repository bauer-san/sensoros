# compute_homography.py
import cv2
import numpy as np
import json

def compute_homography(image_points_file, world_points_file):
    with open(image_points_file) as f:
        img_data = json.load(f)
    with open(world_points_file) as f:
        world_data = json.load(f)

    image_pts = np.array(img_data["image_points"], dtype=np.float32)
    world_pts = np.array(world_data["world_points"], dtype=np.float32)

    assert len(image_pts) == len(world_pts), \
        f"Point count mismatch: {len(image_pts)} image vs {len(world_pts)} world"
    assert len(image_pts) >= 4, \
        "Need at least 4 point correspondences"

    # RANSAC mode filters outliers if any points were
    # clicked or measured inaccurately
    H, mask = cv2.findHomography(
        image_pts, world_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=0.5   # meters — tighten if points are accurate
    )

    inliers = mask.sum()
    print(f"Homography computed from {inliers}/{len(image_pts)} inlier points")

    if inliers < 4:
        print("WARNING: Too few inliers — check your point measurements")
        return None

    return H

def validate_homography(H, image_points, world_points):
    """
    Reproject each calibration point through H and measure error.
    This tells you how accurate your homography is at known locations.
    """
    print("\nReprojection errors at calibration points:")
    print(f"{'Point':<8} {'Image px':<20} {'World (m)':<20} "
          f"{'Projected (m)':<20} {'Error (m)':<10}")
    print("-" * 78)

    errors = []
    for i, (img_pt, world_pt) in enumerate(zip(image_points, world_points)):
        # Project image point through homography
        pt = np.array([[img_pt]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, H)[0][0]

        error = np.linalg.norm(projected - np.array(world_pt))
        errors.append(error)

        print(
            f"P{i+1:<6} "
            f"({img_pt[0]:.0f}, {img_pt[1]:.0f}){'':<8} "
            f"({world_pt[0]:.2f}, {world_pt[1]:.2f}){'':<10} "
            f"({projected[0]:.2f}, {projected[1]:.2f}){'':<8} "
            f"{error:.3f}m"
        )

    print(f"\nMean error:  {np.mean(errors):.3f}m")
    print(f"Max error:   {np.max(errors):.3f}m")
    print(f"Std dev:     {np.std(errors):.3f}m")

    if np.mean(errors) < 0.15:
        print("✅ Excellent calibration")
    elif np.mean(errors) < 0.40:
        print("✅ Good calibration — adequate for zone assignment")
    elif np.mean(errors) < 1.0:
        print("⚠️  Marginal — consider adding more calibration points")
    else:
        print("❌ Poor calibration — recheck measurements and point clicks")

    return errors

def save_calibration(H, metadata, output_file="calibration.json"):
    output = {
        "homography_matrix": H.tolist(),
        "metadata": metadata
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nCalibration saved to {output_file}")

def main():
    # Load points
    with open("image_points.json") as f:
        image_pts = np.array(json.load(f)["image_points"], dtype=np.float32)
    with open("world_points.json") as f:
        world_data = json.load(f)
        world_pts = np.array(world_data["world_points"], dtype=np.float32)

    # Compute
    H = compute_homography("image_points.json", "world_points.json")
    if H is None:
        return

    # Validate
    validate_homography(H, image_pts, world_pts)

    # Save
    save_calibration(H, {
        "origin": world_data["notes"]["origin"],
        "x_axis": world_data["notes"]["x_axis"],
        "y_axis": world_data["notes"]["y_axis"],
        "units": "meters",
        "image_resolution": [1920, 1080],
        "num_calibration_points": len(image_pts)
    })

if __name__ == "__main__":
    main()