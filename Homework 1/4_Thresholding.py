# ================================================================
# GRAYSCALE AND THRESHOLDING COMPARISON
# ================================================================
#
# For every image in the input folder, this program:
#
#   1. Converts the image to grayscale.
#   2. Applies simple/global thresholding.
#   3. Applies adaptive thresholding.
#   4. Places all three images side-by-side.
#   5. Adds labels.
#   6. Saves the comparison as one image.
#
# Required packages:
#
#   py -m pip install opencv-python numpy
#
# ================================================================


import cv2
import numpy as np
from pathlib import Path


# ================================================================
# USER SETTINGS
# ================================================================

# Folder containing your source images.
INPUT_FOLDER = Path(
    r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\Thresholding\Source"
)

# Folder where the finished comparison images will be saved.
OUTPUT_FOLDER = Path(
    r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\Thresholding\Output"
)


# ------------------------------------------------
# SIMPLE THRESHOLD VALUE
# ------------------------------------------------
#
# Pixels BELOW this value become black.
# Pixels ABOVE this value become white.
#
# Pixel brightness ranges from:
#
#       0   = black
#       255 = white
#

SIMPLE_THRESHOLD = 127


# ------------------------------------------------
# ADAPTIVE THRESHOLD SETTINGS
# ------------------------------------------------
#
# BLOCK_SIZE is the neighborhood OpenCV examines when calculating
# the local threshold.
#
# It must be an odd number such as:
#
#       3, 5, 7, 9, 11, 15, 21, etc.
#

ADAPTIVE_BLOCK_SIZE = 11


# C is subtracted from the locally calculated threshold.
#
# Changing this value changes how aggressively the image is
# separated into black and white.

ADAPTIVE_C = 2

# ================================================================
# FIND ALL IMAGES
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




print(f"Found {len(image_files)} image(s).")


# ================================================================
# FUNCTION TO ADD A LABEL ABOVE AN IMAGE
# ================================================================

def add_label(image, text):

    # Thresholded images contain only one grayscale channel.
    #
    # We convert them to 3-channel images so that they can easily
    # be joined together using np.hstack().
    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )


    # Create a white area above the image for the text label.

    label_height = 50

    label_area = np.full(
        (
            label_height,
            image.shape[1],
            3
        ),
        255,
        dtype=np.uint8
    )


    # Write the label.

    cv2.putText(
        label_area,
        text,

        # Location of text
        (15, 33),

        # Font
        cv2.FONT_HERSHEY_SIMPLEX,

        # Font size
        0.8,

        # Black text
        (0, 0, 0),

        # Text thickness
        2,

        cv2.LINE_AA
    )


    # Put the label above the image.

    labeled_image = np.vstack(
        (
            label_area,
            image
        )
    )


    return labeled_image


# ================================================================
# PROCESS EACH IMAGE
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
    # LOAD IMAGE
    # ------------------------------------------------------------

    image = cv2.imread(
        str(image_file)
    )


    if image is None:

        print("Could not read image. Skipping.")

        continue


    # ------------------------------------------------------------
    # CONVERT TO GRAYSCALE
    # ------------------------------------------------------------

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ============================================================
    # SIMPLE / GLOBAL THRESHOLD
    # ============================================================
    #
    # Every pixel is compared to the SAME threshold.
    #
    # For example, with a threshold of 127:
    #
    #       pixel <= 127  --> black
    #
    #       pixel > 127   --> white
    #

    _, simple_threshold = cv2.threshold(

        grayscale,

        SIMPLE_THRESHOLD,

        # White pixel value
        255,

        cv2.THRESH_BINARY
    )


    # ============================================================
    # ADAPTIVE THRESHOLD
    # ============================================================
    #
    # Adaptive thresholding calculates a DIFFERENT threshold for
    # different areas of the image.
    #
    # This can work better when lighting is uneven.
    #
    # ADAPTIVE_THRESH_GAUSSIAN_C means nearby pixels are weighted
    # using a Gaussian distribution.
    #

    adaptive_threshold = cv2.adaptiveThreshold(

        grayscale,

        # Output white value
        255,

        # Calculate threshold using Gaussian-weighted neighborhood
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        # Produce a black-and-white binary image
        cv2.THRESH_BINARY,

        # Neighborhood size
        ADAPTIVE_BLOCK_SIZE,

        # Constant subtracted from calculated threshold
        ADAPTIVE_C
    )


    # ============================================================
    # ADD LABELS
    # ============================================================

    original_labeled = add_label(
    image,
    "Original"
)


    simple_labeled = add_label(
        simple_threshold,
        f"Simple Threshold = {SIMPLE_THRESHOLD}"
    )


    adaptive_labeled = add_label(
        adaptive_threshold,
        "Adaptive Threshold"
    )


    # ============================================================
    # COMBINE INTO 1 x 3 IMAGE
    # ============================================================
    #
    # np.hstack() means "horizontal stack."
    #
    # The final arrangement is:
    #
    #   Grayscale | Simple | Adaptive
    #

    comparison = np.hstack(
        (
            original_labeled,
            simple_labeled,
            adaptive_labeled
        )
    )


    # ============================================================
    # SAVE RESULT
    # ============================================================

    output_filename = (
        image_file.stem
        + "_threshold_comparison.jpg"
    )


    output_path = (
        OUTPUT_FOLDER
        / output_filename
    )


    cv2.imwrite(
        str(output_path),
        comparison
    )


    print(
        f"Saved: {output_path}"
    )

    simple_filename = (
        image_file.stem
        + "_simple_threshold.jpg"
    )

    simple_path = (
        OUTPUT_FOLDER
        / simple_filename
    )

    cv2.imwrite(
        str(simple_path),
        simple_threshold
    )


    # ------------------------------------------------------------
    # 3. SAVE ADAPTIVE THRESHOLD IMAGE BY ITSELF
    # ------------------------------------------------------------

    adaptive_filename = (
        image_file.stem
        + "_adaptive_threshold.jpg"
    )

    adaptive_path = (
        OUTPUT_FOLDER
        / adaptive_filename
    )

    cv2.imwrite(
        str(adaptive_path),
        adaptive_threshold
    )
# ================================================================
# FINISHED
# ================================================================

print()
print("Finished.")
print("Results saved in:")
print(OUTPUT_FOLDER)