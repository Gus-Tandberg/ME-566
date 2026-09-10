import cv2
import numpy as np
import os

# ============================================================
# USER SETTINGS
# ============================================================

CALIBRATION_FILE = (
    r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\HW1_calibration_images\bulk\camera_calibration.npz"
)

VALIDATION_IMAGE = (
    r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\HW1_calibration_images\Test image\image_0.jpg"
)

OUTPUT_IMAGE = (
    r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\HW1_calibration_images\bulk\result.jpg"
)


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

calibration = np.load(CALIBRATION_FILE)

camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["distortion_coefficients"]

# Read checkerboard information from calibration file
CHECKERBOARD = tuple(
    calibration["checkerboard_internal_corners"].astype(int)
)

square_size = float(calibration["square_size"])

# Original calibration error information
rms_error = float(calibration["rms_error"])
overall_reprojection_error = float(
    calibration["overall_reprojection_error"]
)

per_image_errors = calibration["per_image_errors"]


# ============================================================
# PRINT CALIBRATION INFORMATION
# ============================================================

print("\n========================================")
print("CAMERA CALIBRATION INFORMATION")
print("========================================")

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)

print(f"\nCheckerboard internal corners: {CHECKERBOARD}")
print(f"Checkerboard square size: {square_size}")

print(f"\nCalibration RMS error:")
print(f"{rms_error:.4f} pixels")

print("\nOverall calibration reprojection error:")
print(f"{overall_reprojection_error:.4f} pixels")

print("\nPer-image calibration errors:")

for i, error in enumerate(per_image_errors):
    print(f"Image {i + 1}: {float(error):.4f} pixels")


# ============================================================
# CREATE 3D CHECKERBOARD POINTS
# ============================================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= square_size


# ============================================================
# LOAD UNUSED VALIDATION IMAGE
# ============================================================

image = cv2.imread(VALIDATION_IMAGE)

if image is None:
    print("\nERROR: Could not load validation image.")
    print(VALIDATION_IMAGE)
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# CHECK IMAGE SIZE
# ============================================================

calibration_width = int(calibration["image_width"])
calibration_height = int(calibration["image_height"])

height, width = gray.shape

print("\n========================================")
print("VALIDATION IMAGE")
print("========================================")

print(f"Calibration image size: {calibration_width} x {calibration_height}")
print(f"Validation image size:  {width} x {height}")

if width != calibration_width or height != calibration_height:
    print("\nWARNING:")
    print("The validation image resolution does not match the")
    print("resolution used during camera calibration.")


# ============================================================
# FIND CHECKERBOARD CORNERS
# ============================================================

found, corners = cv2.findChessboardCorners(
    gray,
    CHECKERBOARD,
    flags=(
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
)

if not found:
    print("\nERROR: Checkerboard could not be detected.")
    exit()


# ============================================================
# REFINE CORNER LOCATIONS
# ============================================================

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

corners_refined = cv2.cornerSubPix(
    gray,
    corners,
    (11, 11),
    (-1, -1),
    criteria
)


# ============================================================
# ESTIMATE CHECKERBOARD POSE
# ============================================================

success, rvec, tvec = cv2.solvePnP(
    objp,
    corners_refined,
    camera_matrix,
    dist_coeffs
)

if not success:
    print("\nERROR: solvePnP failed.")
    exit()


# ============================================================
# PROJECT 3D CHECKERBOARD POINTS BACK ONTO IMAGE
# ============================================================

projected_points, _ = cv2.projectPoints(
    objp,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs
)


# ============================================================
# CALCULATE VALIDATION REPROJECTION ERROR
# ============================================================
# Convert both arrays to simple N x 2 arrays
detected = corners_refined.reshape(-1, 2).astype(np.float64)
projected = projected_points.reshape(-1, 2).astype(np.float64)

# Error vector for each checkerboard corner
differences = detected - projected

# Euclidean error of each corner, in pixels
point_errors = np.linalg.norm(differences, axis=1)

# RMS reprojection error
validation_error = np.sqrt(
    np.mean(point_errors ** 2)
)

# Additional useful statistics
mean_point_error = np.mean(point_errors)
max_point_error = np.max(point_errors)

# ============================================================
# PRINT VALIDATION RESULTS
# ============================================================

print("\n========================================")
print("VALIDATION RESULTS")
print("========================================")

print(
    f"\nValidation reprojection error: "
    f"{validation_error:.4f} pixels"
)

print(
    f"Original calibration reprojection error: "
    f"{overall_reprojection_error:.4f} pixels"
)

print(
    f"Mean individual corner error: "
    f"{mean_point_error:.4f} pixels"
)

print(
    f"Maximum individual corner error: "
    f"{max_point_error:.4f} pixels"
)


# ============================================================
# COMPARE VALIDATION ERROR TO CALIBRATION ERROR
# ============================================================

difference = validation_error - overall_reprojection_error

print("\nDifference from calibration error:")
print(f"{difference:+.4f} pixels")

if validation_error < 0.5:
    print("\nResult: EXCELLENT validation error.")

elif validation_error < 1.0:
    print("\nResult: GOOD validation error.")

elif validation_error < 2.0:
    print("\nResult: QUESTIONABLE validation error.")

else:
    print("\nResult: POOR validation error.")


# ============================================================
# DRAW DETECTED AND PROJECTED POINTS
# ============================================================

display = image.copy()

# Green = actual detected checkerboard corners
for point in corners_refined:
    x, y = point.ravel()
    cv2.circle(
        display,
        (int(round(x)), int(round(y))),
        6,
        (0, 255, 0),
        -1
    )

# Red = projected points predicted by calibration
for point in projected_points:
    x, y = point.ravel()
    cv2.circle(
        display,
        (int(round(x)), int(round(y))),
        3,
        (0, 0, 255),
        -1
    )


# ============================================================
# ADD TEXT TO RESULT IMAGE
# ============================================================

cv2.putText(
    display,
    f"Validation Error: {validation_error:.4f} px",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.putText(
    display,
    "Green = Detected   Red = Projected",
    (30, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)


# ============================================================
# SAVE RESULT
# ============================================================

cv2.imwrite(OUTPUT_IMAGE, display)

print("\nValidation image saved to:")
print(OUTPUT_IMAGE)


# ============================================================
# DISPLAY RESULT
# ============================================================

# Resize only for display if image is very large.
max_display_width = 1400

if display.shape[1] > max_display_width:

    scale = max_display_width / display.shape[1]

    display_resized = cv2.resize(
        display,
        None,
        fx=scale,
        fy=scale
    )

else:
    display_resized = display


cv2.imshow(
    "Camera Calibration Validation",
    display_resized
)

cv2.waitKey(0)
cv2.destroyAllWindows()