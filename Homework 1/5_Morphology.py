# ================================================================
# BINARY IMAGE MORPHOLOGICAL OPERATIONS
# ================================================================
#
# This program reads binary images from a folder and applies:
#
#   1. Erosion
#   2. Dilation
#   3. Opening
#   4. Closing
#   5. Morphological Gradient
#
# The original image and the five processed images are arranged
# into a labeled 2 x 3 comparison image.
#
# IMPORTANT:
# The source images are already binary, so this program does NOT
# convert them to grayscale or apply thresholding.
#
# ================================================================


import cv2
import numpy as np
from pathlib import Path


# ================================================================
# USER SETTINGS
# ================================================================

# Folder containing your binary source images.
#
# Change this to your actual folder location.

INPUT_FOLDER = Path(
    r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\Morphology\Source"
)


# Folder where the comparison images will be saved.
#
# The program will create this folder if it does not already exist.

OUTPUT_FOLDER = Path(
    r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\Morphology\Output"
)


# ------------------------------------------------
# MORPHOLOGICAL KERNEL SIZE
# ------------------------------------------------
#
# A 3 x 3 kernel means that each operation examines a
# 3-pixel by 3-pixel neighborhood.
#
# You can change this to:
#
#     (5, 5)
#
# if you want a stronger effect.

KERNEL_SIZE = (3, 3)


# Number of times each operation is performed.
#
# 1 is a good starting point.
# Increasing this makes erosion and dilation more aggressive.

ITERATIONS = 1


# Height of the white label above each image.

LABEL_HEIGHT = 65

# ================================================================
# FIND ALL SOURCE IMAGES
# ================================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


image_files = [
    file
    for file in INPUT_FOLDER.iterdir()
    if file.is_file()
    and file.suffix.lower() in VALID_EXTENSIONS
]


image_files.sort()



# ================================================================
# CREATE THE MORPHOLOGICAL KERNEL
# ================================================================
#
# The kernel is the small neighborhood that moves across the image
# while OpenCV performs the morphological operation.
#
# MORPH_RECT creates a rectangular kernel.
#
# A 3 x 3 kernel looks like:
#
#       1  1  1
#       1  1  1
#       1  1  1
#

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    KERNEL_SIZE
)


# ================================================================
# FUNCTION FOR ADDING LABELS
# ================================================================

def add_label(image, title, description):
    """
    Add a white label area above an image.

    title:
        Name of the operation.

    description:
        Short explanation of what the operation does.
    """


    # If the image happens to be stored as a single-channel binary
    # image, convert it to 3 channels ONLY for creating the
    # comparison sheet.
    #
    # This is NOT a grayscale processing step.
    # It simply allows all images and labels to be stacked together.

    if len(image.shape) == 2:

        display_image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    else:

        display_image = image.copy()


    # ------------------------------------------------------------
    # CREATE WHITE LABEL AREA
    # ------------------------------------------------------------

    label_area = np.full(
        (
            LABEL_HEIGHT,
            display_image.shape[1],
            3
        ),
        255,
        dtype=np.uint8
    )


    # ------------------------------------------------------------
    # WRITE OPERATION NAME
    # ------------------------------------------------------------

    cv2.putText(
        label_area,
        title,

        # Text position
        (10, 25),

        cv2.FONT_HERSHEY_SIMPLEX,

        # Font size
        0.65,

        # Black text
        (0, 0, 0),

        # Thickness
        2,

        cv2.LINE_AA
    )


    # ------------------------------------------------------------
    # WRITE SHORT DESCRIPTION
    # ------------------------------------------------------------

    cv2.putText(
        label_area,
        description,

        (10, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (0, 0, 0),

        1,

        cv2.LINE_AA
    )


    # Put the label above the image.

    labeled_image = np.vstack(
        (
            label_area,
            display_image
        )
    )


    return labeled_image


# ================================================================
# PROCESS EVERY IMAGE
# ================================================================

for image_number, image_file in enumerate(
    image_files,
    start=1
):


    print(
        f"Processing {image_number} of {len(image_files)}: "
        f"{image_file.name}"
    )


    # ------------------------------------------------------------
    # LOAD THE ORIGINAL IMAGE
    # ------------------------------------------------------------
    #
    # We are NOT converting it to grayscale or thresholding it.

    original = cv2.imread(
        str(image_file)
    )


    # If OpenCV cannot open the image, skip it.

    if original is None:

        print("Could not read image. Skipping.")
        continue


    # ============================================================
    # 1. EROSION
    # ============================================================
    #
    # Erosion SHRINKS white regions.
    #
    # It can:
    #
    #   - remove small white noise
    #   - thin white objects
    #   - enlarge black gaps
    #

    erosion = cv2.erode(
        original,
        kernel,
        iterations=ITERATIONS
    )


    # ============================================================
    # 2. DILATION
    # ============================================================
    #
    # Dilation EXPANDS white regions.
    #
    # It can:
    #
    #   - thicken white objects
    #   - connect nearby white regions
    #   - fill very small black gaps
    #

    dilation = cv2.dilate(
        original,
        kernel,
        iterations=ITERATIONS
    )


    # ============================================================
    # 3. OPENING
    # ============================================================
    #
    # Opening is:
    #
    #       Erosion followed by Dilation
    #
    # It is useful for removing small WHITE objects or white noise
    # while preserving larger objects.

    opening = cv2.morphologyEx(
        original,
        cv2.MORPH_OPEN,
        kernel,
        iterations=ITERATIONS
    )


    # ============================================================
    # 4. CLOSING
    # ============================================================
    #
    # Closing is:
    #
    #       Dilation followed by Erosion
    #
    # It is useful for:
    #
    #   - filling small BLACK holes
    #   - closing small gaps
    #   - connecting nearby white regions
    #

    closing = cv2.morphologyEx(
        original,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=ITERATIONS
    )


    # ============================================================
    # 5. MORPHOLOGICAL GRADIENT
    # ============================================================
    #
    # The morphological gradient is approximately:
    #
    #       Dilation - Erosion
    #
    # This leaves the boundaries or outlines of objects.

    gradient = cv2.morphologyEx(
        original,
        cv2.MORPH_GRADIENT,
        kernel,
        iterations=ITERATIONS
    )


    # ============================================================
    # ADD LABELS
    # ================================================================

    original_labeled = add_label(
        original,
        "Original",
        "Unmodified binary image"
    )


    erosion_labeled = add_label(
        erosion,
        "Erosion",
        "Shrinks white regions"
    )


    dilation_labeled = add_label(
        dilation,
        "Dilation",
        "Expands white regions"
    )


    opening_labeled = add_label(
        opening,
        "Opening",
        "Removes small white objects"
    )


    closing_labeled = add_label(
        closing,
        "Closing",
        "Fills small black gaps"
    )


    gradient_labeled = add_label(
        gradient,
        "Morphological Gradient",
        "Highlights object boundaries"
    )


    # ============================================================
    # CREATE THE 2 x 3 COMPARISON
    # ============================================================
    #
    # First row:
    #
    #       Original | Erosion | Dilation
    #
    # Second row:
    #
    #       Opening | Closing | Gradient
    #

    row_1 = np.hstack(
        (
            original_labeled,
            erosion_labeled,
            dilation_labeled
        )
    )


    row_2 = np.hstack(
        (
            opening_labeled,
            closing_labeled,
            gradient_labeled
        )
    )


    # Stack the two rows vertically.

    comparison = np.vstack(
        (
            row_1,
            row_2
        )
    )


    # ============================================================
    # SAVE THE COMPARISON IMAGE
    # ================================================================

    output_filename = (
        image_file.stem
        + "_morphology_comparison.jpg"
    )


    output_path = (
        OUTPUT_FOLDER
        / output_filename
    )


    saved = cv2.imwrite(
        str(output_path),
        comparison
    )


    if saved:

        print(
            f"Saved: {output_path}"
        )

    else:

        print(
            "ERROR: Could not save comparison image."
        )


# ================================================================
# FINISHED
# ================================================================

print()
print("Finished.")
print()

print("Results saved in:")
print(OUTPUT_FOLDER)