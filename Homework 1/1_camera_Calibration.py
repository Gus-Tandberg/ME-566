# ================================================================
# CAMERA CALIBRATION USING OPENCV
# ================================================================
#
# CAMERA / LENS:
#   Raspberry Pi camera with 16 mm C-mount telephoto lens
#
# CALIBRATION TARGET:
#   8 squares x 11 squares checkerboard
#
# IMPORTANT:
# OpenCV counts the INTERNAL CORNERS of the checkerboard, not the
# number of black/white squares.
#
# An 8 x 11 SQUARE checkerboard therefore contains:
#
#       7 x 10 INTERNAL CORNERS
#
#
# DISTORTION MODEL:
#
# This program uses OpenCV's standard perspective camera model
# with radial and tangential lens distortion.
#
# This is appropriate for the 16 mm telephoto lens because it is
# a conventional rectilinear lens, not a fisheye lens.
#
# The manufacturer's focal length specification is NOT used to
# calculate the camera calibration.
#
# The following are all determined from the photographs:
#
#       fx
#       fy
#       cx
#       cy
#       k1
#       k2
#       p1
#       p2
#       k3
#
# This allows the camera/lens combination to be calibrated based
# on its ACTUAL optical behavior rather than nominal specifications.
#
#
# Required packages:
#
#       pip install opencv-python numpy
#
# ================================================================


# ================================================================
# IMPORT REQUIRED LIBRARIES
# ================================================================

import cv2

# NumPy handles the arrays and matrices used by OpenCV.
import numpy as np

# glob is used to find every image inside the calibration folder.
import glob

# os helps create file paths that work correctly in Windows.
import os


# ================================================================
# USER SETTINGS
# ================================================================


# ------------------------------------------------
# FOLDER CONTAINING YOUR CALIBRATION IMAGES
# ------------------------------------------------
#
# Replace this placeholder with the actual folder when you run
# the program.
#
# The letter r before the quotation mark is important for Windows
# paths because Windows uses backslashes.
#
# Example:
#
# IMAGE_FOLDER = r"C:\Users\Augustus\Pictures\CameraCalibration"
#

IMAGE_FOLDER = r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\HW1_calibration_images\2"


# ------------------------------------------------
# CHECKERBOARD SIZE
# ------------------------------------------------
#
# Your checkerboard is:
#
#       8 squares across
#       11 squares down
#
# OpenCV wants INTERNAL CORNERS instead:
#
#       7 corners across
#       10 corners down
#

CHECKERBOARD_SIZE = (10, 7)


# ------------------------------------------------
# PHYSICAL CHECKERBOARD SQUARE SIZE
# ------------------------------------------------
#
# Measure ONE square on your printed/physical checkerboard.
#
# You can use any unit:
#
#       millimeters
#       centimeters
#       inches
#
# Just stay consistent.
#
# Example:
#
# If each square is 25 mm x 25 mm:
#
#       SQUARE_SIZE = 25.0
#
# The square size does NOT determine the focal length or lens
# distortion.
#
# It establishes the real-world scale of the checkerboard and is
# important if you later use the calibration for pose or distance
# measurements.
#

SQUARE_SIZE = 23.38


# ================================================================
# CREATE THE KNOWN 3D CHECKERBOARD POINTS
# ================================================================
#
# Camera calibration works by comparing:
#
#   1. Where each checkerboard corner is known to exist on the
#      physical calibration board.
#
#   2. Where that corner appears in the photograph.
#
#
# Because the checkerboard is flat, we define the checkerboard as
# lying in the X-Y plane.
#
# Therefore:
#
#       Z = 0
#
# for every checkerboard corner.
#
#
# The coordinates look like:
#
#       (0, 0, 0)
#       (1, 0, 0)
#       (2, 0, 0)
#       ...
#
#       (0, 1, 0)
#       (1, 1, 0)
#       ...
#
#
# Those coordinates are then multiplied by SQUARE_SIZE so that the
# point spacing corresponds to your physical checkerboard.
# ================================================================


corners_horizontal = CHECKERBOARD_SIZE[0]
corners_vertical = CHECKERBOARD_SIZE[1]


# Create an empty array.
#
# Your board contains:
#
#       7 * 10 = 70
#
# internal corners.
#
# Each point contains an X, Y, and Z coordinate.
#
# Therefore this creates a:
#
#       70 x 3
#
# array.
#

object_points_for_one_image = np.zeros(
    (corners_horizontal * corners_vertical, 3),
    dtype=np.float32
)


# Generate the X-Y grid.
object_points_for_one_image[:, :2] = np.mgrid[
    0:corners_horizontal,
    0:corners_vertical
].T.reshape(-1, 2)


# Convert the spacing from arbitrary "1 unit" spacing into the
# physical checkerboard square size.
object_points_for_one_image *= SQUARE_SIZE


# ================================================================
# STORAGE FOR CALIBRATION DATA
# ================================================================
#
# object_points:
#
#       Known physical checkerboard coordinates.
#
# image_points:
#
#       Locations of those corners measured in the photographs.
#
# image_names_used:
#
#       Keeps track of which photographs successfully contributed
#       to the calibration.
#

object_points = []

image_points = []

image_names_used = []


# ================================================================
# FIND ALL IMAGES IN THE FOLDER
# ================================================================

image_extensions = [

    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.tif",
    "*.tiff",

    # Include uppercase versions because some cameras/software
    # may save files with uppercase extensions.
    "*.JPG",
    "*.JPEG",
    "*.PNG",
    "*.BMP",
    "*.TIF",
    "*.TIFF"
]


image_files = []


# Search for every supported image format.
for extension in image_extensions:

    search_path = os.path.join(
        IMAGE_FOLDER,
        extension
    )

    image_files.extend(
        glob.glob(search_path)
    )


# Remove any duplicate paths and sort alphabetically.
image_files = sorted(set(image_files))


# ================================================================
# MAKE SURE IMAGES ACTUALLY EXIST
# ================================================================

if len(image_files) == 0:

    print()
    print("================================================")
    print("ERROR")
    print("================================================")
    print()

    print("No calibration images were found.")
    print()

    print("Check IMAGE_FOLDER:")
    print(IMAGE_FOLDER)
    print()

    raise SystemExit


print()
print("================================================")
print("CAMERA CALIBRATION")
print("================================================")
print()

print(f"Calibration images found: {len(image_files)}")
print()

print("Checkerboard:")
print("  8 x 11 squares")
print("  7 x 10 internal corners")
print()

print("Camera model:")
print("  Standard perspective / pinhole model")
print("  Radial + tangential distortion")
print()

print("Searching for checkerboard corners...")
print()


# ================================================================
# PROCESS ALL CALIBRATION IMAGES
# ================================================================


# All calibration images should have the same resolution.
#
# This variable will store the resolution of the first valid image.
image_size = None


for image_number, image_file in enumerate(image_files, start=1):

    print(
        f"[{image_number}/{len(image_files)}] "
        f"{os.path.basename(image_file)}"
    )


    # ------------------------------------------------------------
    # LOAD THE IMAGE
    # ------------------------------------------------------------

    image = cv2.imread(image_file)


    if image is None:

        print("    ERROR: Image could not be opened.")
        print("    Skipping this image.")
        print()

        continue


    # ------------------------------------------------------------
    # DETERMINE IMAGE RESOLUTION
    # ------------------------------------------------------------

    current_size = (
        image.shape[1],     # width
        image.shape[0]      # height
    )


    # If this is the first valid image, remember its resolution.
    if image_size is None:

        image_size = current_size

        print(
            f"    Calibration resolution: "
            f"{image_size[0]} x {image_size[1]}"
        )


    # Every other image must match the first image.
    elif current_size != image_size:

        print(
            "    WARNING: Image resolution does not match the "
            "other calibration images."
        )

        print(
            f"    Expected: {image_size[0]} x {image_size[1]}"
        )

        print(
            f"    Found:    {current_size[0]} x {current_size[1]}"
        )

        print("    Skipping this image.")
        print()

        continue


    # ------------------------------------------------------------
    # CONVERT TO GRAYSCALE
    # ------------------------------------------------------------
    #
    # Checkerboard detection only needs image brightness.
    #

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ------------------------------------------------------------
    # FIND THE CHECKERBOARD
    # ------------------------------------------------------------
    #
    # findChessboardCornersSB() is the newer "sector based"
    # checkerboard detector in OpenCV.
    #
    # It is generally more accurate and robust than the older
    # findChessboardCorners() function, especially for high
    # resolution calibration photographs.
    #
    # "SB" stands for sector-based.
    #

    checkerboard_found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD_SIZE,

        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            +
            cv2.CALIB_CB_EXHAUSTIVE
            +
            cv2.CALIB_CB_ACCURACY
        )
    )


    # ------------------------------------------------------------
    # IF CHECKERBOARD WAS FOUND
    # ------------------------------------------------------------

    if checkerboard_found:

        print("    Checkerboard FOUND.")


        # Add the known physical checkerboard locations.
        object_points.append(
            object_points_for_one_image.copy()
        )


        # Add the locations measured in this photograph.
        image_points.append(
            corners.astype(np.float32)
        )


        # Remember which photograph this came from.
        image_names_used.append(
            os.path.basename(image_file)
        )


        # --------------------------------------------------------
        # DRAW CORNERS FOR VISUAL INSPECTION
        # --------------------------------------------------------

        preview = image.copy()


        cv2.drawChessboardCorners(
            preview,
            CHECKERBOARD_SIZE,
            corners,
            checkerboard_found
        )


        # Large 10-megapixel images will probably be too large to
        # comfortably fit on the computer screen.
        #
        # We therefore create a smaller preview.
        #
        # IMPORTANT:
        #
        # Only the DISPLAY is resized.
        #
        # The full-resolution corner coordinates are still used
        # for the actual calibration.
        #

        max_preview_width = 1200


        if preview.shape[1] > max_preview_width:

            display_scale = (
                max_preview_width /
                preview.shape[1]
            )


            preview = cv2.resize(
                preview,
                None,
                fx=display_scale,
                fy=display_scale,
                interpolation=cv2.INTER_AREA
            )


        cv2.imshow(
            "Detected Calibration Checkerboard",
            preview
        )


        # Display for 300 milliseconds.
        #
        # Change this number if you want to inspect the detected
        # corners for longer.
        #
        # For example:
        #
        #       1000 = 1 second
        #

        cv2.waitKey(300)


    else:

        print("    Checkerboard NOT found.")


    print()


# Close the OpenCV preview window.
cv2.destroyAllWindows()


# ================================================================
# CHECK HOW MANY IMAGES WERE SUCCESSFULLY DETECTED
# ================================================================

number_successful = len(image_points)


print()
print("================================================")
print("CHECKERBOARD DETECTION SUMMARY")
print("================================================")
print()

print(f"Total images:      {len(image_files)}")
print(f"Usable images:     {number_successful}")
print(f"Rejected images:   {len(image_files) - number_successful}")
print()


# Technically calibration can be performed from only a few views,
# but that does not mean it will be a GOOD calibration.
#
# For this camera I recommend substantially more than the minimum.
#

if number_successful < 2:

    print("ERROR:")
    print("Too few images contain a successfully detected checkerboard.")
    print()

    print("Check:")
    print("  - Checkerboard dimensions")
    print("  - Image focus")
    print("  - Lighting")
    print("  - Motion blur")
    print("  - Whether the entire checkerboard is visible")
    print()

    raise SystemExit





# ================================================================
# CAMERA CALIBRATION
# ================================================================
#
# We are now asking OpenCV to solve for the camera's intrinsic
# parameters.
#
#
# CAMERA MATRIX
# -------------
#
# The intrinsic camera matrix is:
#
#
#           [ fx   0   cx ]
#     K  =  [  0  fy   cy ]
#           [  0   0    1 ]
#
#
# fx:
#     Effective horizontal focal length measured in PIXELS.
#
# fy:
#     Effective vertical focal length measured in PIXELS.
#
# cx:
#     X-coordinate of the optical center / principal point.
#
# cy:
#     Y-coordinate of the optical center / principal point.
#
#
# DISTORTION
# ----------
#
# We will use the standard five-coefficient OpenCV model:
#
#       k1
#       k2
#       p1
#       p2
#       k3
#
#
# k1, k2, k3:
#
#       Radial distortion.
#
#       These account mainly for barrel or pincushion distortion.
#
#
# p1, p2:
#
#       Tangential distortion.
#
#       These account for imperfect alignment between the lens
#       elements and the image sensor.
#
#
# We are intentionally NOT using:
#
#       cv2.fisheye.calibrate()
#
# because this 16 mm telephoto lens is a conventional rectilinear
# lens rather than an extreme-wide-angle/fisheye optic.
#
#
# We are also NOT supplying the advertised 16 mm focal length to
# OpenCV.
#
# The focal length in pixels is estimated entirely from the
# checkerboard photographs.
# ================================================================


# Keep the standard five-parameter distortion model.
#
# No special calibration flags are required here.
calibration_flags = 0


rms_error, \
camera_matrix, \
distortion_coefficients, \
rotation_vectors, \
translation_vectors = cv2.calibrateCamera(

    object_points,

    image_points,

    image_size,

    None,

    None,

    flags=calibration_flags
)


# ================================================================
# EXTRACT INDIVIDUAL CAMERA PARAMETERS
# ================================================================

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]

cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]


# Flatten the distortion array so it is easier to work with.
dist = distortion_coefficients.flatten()


k1 = dist[0]
k2 = dist[1]

p1 = dist[2]
p2 = dist[3]

k3 = dist[4]


# ================================================================
# CALCULATE REPROJECTION ERROR FOR EACH IMAGE
# ================================================================
#
# Reprojection error is an important measure of calibration
# quality.
#
#
# OpenCV knows:
#
#       1. The physical checkerboard locations.
#
#       2. The camera calibration.
#
#       3. The position/orientation of the checkerboard in each
#          photograph.
#
#
# Using this information, OpenCV can predict where each corner
# SHOULD appear in the photograph.
#
# We compare those predictions with where OpenCV ACTUALLY found
# each corner.
#
# The difference is the reprojection error.
#
#
# The error is measured in PIXELS.
#
# Smaller is better.
# ================================================================


per_image_errors = []

total_squared_error = 0
total_number_of_points = 0


for i in range(len(object_points)):

    # Calculate where the checkerboard points should appear.
    projected_points, _ = cv2.projectPoints(

        object_points[i],

        rotation_vectors[i],

        translation_vectors[i],

        camera_matrix,

        distortion_coefficients
    )


    # Difference between detected and predicted locations.
    point_errors = (
        image_points[i].reshape(-1, 2)
        -
        projected_points.reshape(-1, 2)
    )


    # Find the Euclidean distance for every checkerboard corner.
    distances = np.linalg.norm(
        point_errors,
        axis=1
    )


    # RMS error for this particular photograph.
    image_rms_error = np.sqrt(
        np.mean(distances ** 2)
    )


    per_image_errors.append(
        image_rms_error
    )


    # Add these errors to the overall error calculation.
    total_squared_error += np.sum(
        distances ** 2
    )

    total_number_of_points += len(
        distances
    )


# Overall RMS reprojection error calculated directly from all
# individual points.
overall_reprojection_error = np.sqrt(
    total_squared_error /
    total_number_of_points
)


# ================================================================
# PRINT CALIBRATION RESULTS
# ================================================================

print()
print("================================================")
print("CAMERA CALIBRATION RESULTS")
print("================================================")
print()


print("Image resolution:")
print(
    f"{image_size[0]} x {image_size[1]} pixels"
)
print()


print("Images used:")
print(number_successful)
print()


print("OpenCV RMS calibration error:")
print(f"{rms_error:.6f} pixels")
print()


print("Calculated overall reprojection error:")
print(f"{overall_reprojection_error:.6f} pixels")
print()


# ------------------------------------------------
# CAMERA MATRIX
# ------------------------------------------------

print("Camera matrix:")
print()

print(camera_matrix)
print()


print("Intrinsic camera parameters:")
print()

print(f"fx = {fx:.6f} pixels")
print(f"fy = {fy:.6f} pixels")

print()

print(f"cx = {cx:.6f} pixels")
print(f"cy = {cy:.6f} pixels")

print()


# ------------------------------------------------
# DISTORTION COEFFICIENTS
# ------------------------------------------------

print("Distortion coefficients:")
print()

print(f"k1 = {k1:.10f}")
print(f"k2 = {k2:.10f}")

print()

print(f"p1 = {p1:.10f}")
print(f"p2 = {p2:.10f}")

print()

print(f"k3 = {k3:.10f}")
print()


# ================================================================
# PRINT ERROR FOR EVERY CALIBRATION IMAGE
# ================================================================
#
# This is very useful for identifying individual photographs that
# may be hurting the calibration.
#
# A photograph with a much larger error than the others may have:
#
#       motion blur
#       poor focus
#       glare
#       checkerboard detection error
#       excessive board bending
#
# ================================================================

print()
print("================================================")
print("REPROJECTION ERROR BY IMAGE")
print("================================================")
print()


for image_name, error in zip(
    image_names_used,
    per_image_errors
):

    print(
        f"{error:8.4f} pixels    {image_name}"
    )


print()


# Find the worst image.
worst_index = int(
    np.argmax(per_image_errors)
)


print("Highest-error calibration image:")

print(
    image_names_used[worst_index]
)

print(
    f"Error = "
    f"{per_image_errors[worst_index]:.4f} pixels"
)

print()


# ================================================================
# SAVE CALIBRATION RESULTS
# ================================================================
#
# NumPy's .npz format allows several matrices and values to be
# conveniently saved together.
#
# Another Python/OpenCV program can later load this file without
# needing to recalibrate the camera.
# ================================================================


output_file = os.path.join(
    IMAGE_FOLDER,
    "camera_calibration.npz"
)


np.savez(

    output_file,

    # Main calibration information
    camera_matrix=camera_matrix,

    distortion_coefficients=distortion_coefficients,


    # Individual intrinsic parameters
    fx=fx,
    fy=fy,

    cx=cx,
    cy=cy,


    # Individual distortion parameters
    k1=k1,
    k2=k2,

    p1=p1,
    p2=p2,

    k3=k3,


    # Calibration-quality information
    rms_error=rms_error,

    overall_reprojection_error=overall_reprojection_error,

    per_image_errors=np.array(
        per_image_errors
    ),


    # Image information
    image_width=image_size[0],

    image_height=image_size[1],


    # Calibration target information
    checkerboard_internal_corners=CHECKERBOARD_SIZE,

    checkerboard_squares=(8, 11),

    square_size=SQUARE_SIZE
)


# ================================================================
# FINISHED
# ================================================================



print("Calibration data saved to:")
print()

print(output_file)
print()



print("Calibration finished successfully.")