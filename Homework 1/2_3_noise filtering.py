# =====================================================================
# IMAGE FILTER COMPARISON PROGRAM
# =====================================================================
#
# This program:
#
#   1. Reads every image from an input folder.
#
#   2. Applies:
#        - Gaussian filter:   3x3 and 5x5
#        - Median filter:     3x3 and 5x5
#        - Bilateral filter:  3x3 and 5x5
#        - N-rank filter:     MULTIPLE selectable ranks for 3x3
#        - N-rank filter:     MULTIPLE selectable ranks for 5x5
#
#   3. Places the original and all filtered images into one
#      labeled comparison block.
#
#   4. Saves one comparison block for each original image.
#
#
# REQUIRED PACKAGES:
#
#     py -m pip install opencv-python numpy
#
# =====================================================================


# =====================================================================
# IMPORT LIBRARIES
# =====================================================================

import cv2
import numpy as np

from pathlib import Path


# =====================================================================
# USER SETTINGS
# =====================================================================


# ---------------------------------------------------------------------
# INPUT FOLDER
# ---------------------------------------------------------------------
#
# Change this to the folder containing the original images.
#

INPUT_FOLDER = Path(r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\Noise_filtering")



# ---------------------------------------------------------------------
# OUTPUT FOLDER
# ---------------------------------------------------------------------
#
# The program creates this folder automatically if necessary.
#

OUTPUT_FOLDER = Path(r"C:\Users\Gus\Dropbox\UND School\Fall 2026\ME 566\Homework 1\Processed_noise")


# =====================================================================
# N-RANK SETTINGS
# =====================================================================
#
# You can enter AS MANY rank values as you want.
#
#
# FOR A 3 x 3 WINDOW:
#
# There are 9 pixels total.
#
# Therefore valid values are:
#
#       1 through 9


N_RANKS_3X3 = [
    1,
    4,
    9
]


# ---------------------------------------------------------------------
# 5 x 5 RANK VALUES
# ---------------------------------------------------------------------
#
# A 5 x 5 window contains 25 pixels.
#
# Therefore valid values are:
#
#       1 through 25

N_RANKS_5X5 = [
    1,
    12,
    25
]


# =====================================================================
# BILATERAL FILTER SETTINGS
# =====================================================================


BILATERAL_SIGMA_COLOR = 75

BILATERAL_SIGMA_SPACE = 75


# =====================================================================
# COMPARISON IMAGE SETTINGS
# =====================================================================


# Width of each image tile in the final comparison image.
#
# Filtering occurs BEFORE resizing.
#
# Therefore the filters are still applied to the full-resolution
# original images.

TILE_WIDTH = 500


# Height of the white label area above each image.

LABEL_HEIGHT = 50


# Number of images placed across each row.
#
# For example:
#
#       3
#
# means the comparison image will look like:
#
#       Image  Image  Image
#       Image  Image  Image
#       Image  Image  Image
#       ...
#

TILES_PER_ROW = 3


# =====================================================================
# CHECK FOLDERS AND USER SETTINGS
# =====================================================================


# Make sure the input folder exists.

if not INPUT_FOLDER.exists():

    print()
    print("ERROR:")
    print("The input folder does not exist:")
    print(INPUT_FOLDER)

    raise SystemExit


# Create the output folder if necessary.

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------
# CHECK 3x3 RANK VALUES
# ---------------------------------------------------------------------

for rank in N_RANKS_3X3:

    if rank < 1 or rank > 9:

        print()
        print("ERROR:")
        print(
            f"3x3 rank {rank} is invalid."
        )

        print(
            "3x3 rank values must be between 1 and 9."
        )

        raise SystemExit


# ---------------------------------------------------------------------
# CHECK 5x5 RANK VALUES
# ---------------------------------------------------------------------

for rank in N_RANKS_5X5:

    if rank < 1 or rank > 25:

        print()
        print("ERROR:")
        print(
            f"5x5 rank {rank} is invalid."
        )

        print(
            "5x5 rank values must be between 1 and 25."
        )

        raise SystemExit


# =====================================================================
# FIND ALL IMAGES
# =====================================================================


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


if len(image_files) == 0:

    print()
    print("ERROR:")
    print("No images were found in:")
    print(INPUT_FOLDER)

    raise SystemExit


# =====================================================================
# N-RANK FILTER FUNCTION
# =====================================================================
#
# A rank filter:
#
#   1. Looks at all pixels in a small neighborhood.
#
#   2. Sorts those pixels from smallest to largest.
#
#   3. Chooses the requested ranked value.
#
#
# Example:
#
# Imagine a 3x3 grayscale neighborhood:
#
#        25    80    50
#        20   100    70
#        35    60    40
#
#
# Sorted:
#
#        20
#        25
#        35
#        40
#        50
#        60
#        70
#        80
#       100
#
#
# Rank 1 = 20
# Rank 3 = 35
# Rank 5 = 50
# Rank 9 = 100
#
#
# For color images, this program applies the operation independently
# to the Blue, Green, and Red channels.
# =====================================================================


def rank_filter(image, window_size, rank):

    """
    Apply an N-rank filter to a color image.

    Parameters
    ----------
    image:
        OpenCV image.

    window_size:
        Neighborhood size.

        3 means 3x3.
        5 means 5x5.

    rank:
        Which sorted pixel value should be returned.

        Rank numbering starts at 1.
    """


    # -------------------------------------------------------------
    # CALCULATE NUMBER OF VALUES IN WINDOW
    # -------------------------------------------------------------

    number_of_pixels = window_size * window_size


    # Make sure the requested rank is possible.

    if rank < 1 or rank > number_of_pixels:

        raise ValueError(

            f"Rank {rank} is invalid for a "
            f"{window_size}x{window_size} window."
        )


    # -------------------------------------------------------------
    # BORDER SIZE
    # -------------------------------------------------------------
    #
    # 3x3 requires one pixel around the edges.
    #
    # 5x5 requires two pixels around the edges.
    #

    padding = window_size // 2


    # Split the color image into:
    #
    #     Blue
    #     Green
    #     Red

    channels = cv2.split(image)


    filtered_channels = []


    # -------------------------------------------------------------
    # PROCESS EACH COLOR CHANNEL
    # -------------------------------------------------------------

    for channel in channels:


        # Add a reflected border so that pixels near the outside
        # edge still have a complete neighborhood.

        padded = cv2.copyMakeBorder(

            channel,

            padding,
            padding,
            padding,
            padding,

            cv2.BORDER_REFLECT
        )


        # ---------------------------------------------------------
        # CREATE A WINDOW AROUND EVERY PIXEL
        # ---------------------------------------------------------

        windows = np.lib.stride_tricks.sliding_window_view(

            padded,

            (window_size, window_size)
        )


        # ---------------------------------------------------------
        # FLATTEN EACH WINDOW
        # ---------------------------------------------------------
        #
        # A 3x3 window becomes a list of 9 values.
        #
        # A 5x5 window becomes a list of 25 values.
        #

        windows_flat = windows.reshape(

            windows.shape[0],

            windows.shape[1],

            number_of_pixels
        )


        # Python counts array locations starting with 0.
        #
        # Humans call the smallest value "rank 1."
        #
        # Therefore:
        #
        #     rank 1 -> array index 0
        #
        # That is why we subtract 1.

        rank_index = rank - 1


        # ---------------------------------------------------------
        # FIND REQUESTED RANK
        # ---------------------------------------------------------
        #
        # np.partition() is faster than completely sorting every
        # neighborhood when we only need one particular value.
        #

        selected_pixels = np.partition(

            windows_flat,

            rank_index,

            axis=2

        )[:, :, rank_index]


        selected_pixels = selected_pixels.astype(
            np.uint8
        )


        filtered_channels.append(
            selected_pixels
        )


    # Put Blue, Green, and Red back together.

    filtered_image = cv2.merge(
        filtered_channels
    )


    return filtered_image


# =====================================================================
# MAKE A LABELED IMAGE TILE
# =====================================================================


def make_labeled_tile(image, label):


    # Read the original dimensions.

    original_height = image.shape[0]

    original_width = image.shape[1]


    # Calculate the amount of resizing needed.

    scale = TILE_WIDTH / original_width


    new_height = int(
        original_height * scale
    )


    # Resize while preserving aspect ratio.

    resized = cv2.resize(

        image,

        (TILE_WIDTH, new_height),

        interpolation=cv2.INTER_AREA
    )


    # -------------------------------------------------------------
    # MAKE WHITE LABEL AREA
    # -------------------------------------------------------------

    label_area = np.full(

        (
            LABEL_HEIGHT,
            TILE_WIDTH,
            3
        ),

        255,

        dtype=np.uint8
    )


    # -------------------------------------------------------------
    # DRAW LABEL TEXT
    # -------------------------------------------------------------

    cv2.putText(

        label_area,

        label,

        (12, 33),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (0, 0, 0),

        2,

        cv2.LINE_AA
    )


    # Put the label above the image.

    tile = np.vstack(
        (
            label_area,
            resized
        )
    )


    return tile


# =====================================================================
# BUILD COMPARISON GRID
# =====================================================================
#
# Because the number of rank-filter images can now change, we cannot
# assume that the comparison block will always be 3x3.
#
# This function automatically creates as many rows as necessary.
# =====================================================================


def build_image_grid(tiles, tiles_per_row):


    # Every tile should have the same size because they were all
    # created with make_labeled_tile().

    tile_height = tiles[0].shape[0]

    tile_width = tiles[0].shape[1]


    # -------------------------------------------------------------
    # ADD BLANK TILES IF FINAL ROW IS NOT FULL
    # -------------------------------------------------------------
    #
    # Example:
    #
    # If there are 11 results and we want 3 per row:
    #
    #     Row 1 = 3
    #     Row 2 = 3
    #     Row 3 = 3
    #     Row 4 = 2
    #
    # We add one blank tile so that Row 4 still has the same width.
    #

    while len(tiles) % tiles_per_row != 0:

        blank_tile = np.full(

            (
                tile_height,
                tile_width,
                3
            ),

            255,

            dtype=np.uint8
        )


        tiles.append(
            blank_tile
        )


    # -------------------------------------------------------------
    # CREATE EACH ROW
    # -------------------------------------------------------------

    rows = []


    for start_position in range(
        0,
        len(tiles),
        tiles_per_row
    ):


        row_tiles = tiles[

            start_position:

            start_position + tiles_per_row
        ]


        row = np.hstack(
            row_tiles
        )


        rows.append(
            row
        )


    # -------------------------------------------------------------
    # STACK ALL ROWS
    # -------------------------------------------------------------

    grid = np.vstack(
        rows
    )


    return grid


# =====================================================================
# PROCESS EVERY IMAGE
# =====================================================================


print()
print("============================================================")
print("IMAGE FILTER COMPARISON")
print("============================================================")
print()

print(
    f"Found {len(image_files)} image(s)."
)

print()

print(
    "3x3 N-ranks:",
    N_RANKS_3X3
)

print(
    "5x5 N-ranks:",
    N_RANKS_5X5
)

print()


for image_number, image_file in enumerate(
    image_files,
    start=1
):


    print()
    print("------------------------------------------------------------")

    print(
        f"Processing {image_number} "
        f"of {len(image_files)}"
    )

    print(
        image_file.name
    )


    # =================================================================
    # READ ORIGINAL IMAGE
    # =================================================================

    original = cv2.imread(
        str(image_file)
    )


    if original is None:

        print(
            "ERROR: OpenCV could not read this image."
        )

        continue


    print(
        f"Resolution: "
        f"{original.shape[1]} x {original.shape[0]}"
    )


    # =================================================================
    # STANDARD FILTERS
    # =================================================================


    # Gaussian 3x3

    gaussian_3x3 = cv2.GaussianBlur(

        original,

        (3, 3),

        0
    )


    # Gaussian 5x5

    gaussian_5x5 = cv2.GaussianBlur(

        original,

        (5, 5),

        0
    )


    # Median 3x3

    median_3x3 = cv2.medianBlur(

        original,

        3
    )


    # Median 5x5

    median_5x5 = cv2.medianBlur(

        original,

        5
    )


    # Bilateral 3x3

    bilateral_3x3 = cv2.bilateralFilter(

        original,

        3,

        BILATERAL_SIGMA_COLOR,

        BILATERAL_SIGMA_SPACE
    )


    # Bilateral 5x5

    bilateral_5x5 = cv2.bilateralFilter(

        original,

        5,

        BILATERAL_SIGMA_COLOR,

        BILATERAL_SIGMA_SPACE
    )


    # =================================================================
    # CREATE LIST OF TILES
    # =================================================================
    #
    # We start with the original image and the standard filters.
    #
    # The N-rank results will then be added automatically.
    # =================================================================


    tiles = []


    tiles.append(

        make_labeled_tile(

            original,

            "Original"
        )
    )


    tiles.append(

        make_labeled_tile(

            gaussian_3x3,

            "Gaussian - 3x3"
        )
    )


    tiles.append(

        make_labeled_tile(

            gaussian_5x5,

            "Gaussian - 5x5"
        )
    )


    tiles.append(

        make_labeled_tile(

            median_3x3,

            "Median - 3x3"
        )
    )


    tiles.append(

        make_labeled_tile(

            median_5x5,

            "Median - 5x5"
        )
    )


    tiles.append(

        make_labeled_tile(

            bilateral_3x3,

            "Bilateral - 3x3"
        )
    )


    tiles.append(

        make_labeled_tile(

            bilateral_5x5,

            "Bilateral - 5x5"
        )
    )


    # =================================================================
    # MULTIPLE 3x3 N-RANK FILTERS
    # =================================================================
    #
    # This loop runs once for every value in:
    #
    #       N_RANKS_3X3
    #
    # Therefore:
    #
    #       [2, 4, 7]
    #
    # creates:
    #
    #       Rank 2 - 3x3
    #       Rank 4 - 3x3
    #       Rank 7 - 3x3
    # =================================================================


    for rank in N_RANKS_3X3:


        print(
            f"    Applying 3x3 rank {rank}..."
        )


        filtered_image = rank_filter(

            original,

            window_size=3,

            rank=rank
        )


        tiles.append(

            make_labeled_tile(

                filtered_image,

                f"N-Rank {rank} - 3x3"
            )
        )


    # =================================================================
    # MULTIPLE 5x5 N-RANK FILTERS
    # =================================================================


    for rank in N_RANKS_5X5:


        print(
            f"    Applying 5x5 rank {rank}..."
        )


        filtered_image = rank_filter(

            original,

            window_size=5,

            rank=rank
        )


        tiles.append(

            make_labeled_tile(

                filtered_image,

                f"N-Rank {rank} - 5x5"
            )
        )


    # =================================================================
    # CREATE COMPARISON GRID
    # =================================================================


    comparison_block = build_image_grid(

        tiles,

        TILES_PER_ROW
    )


    # =================================================================
    # ADD ORIGINAL FILENAME AT TOP
    # =================================================================


    title_height = 70


    title_area = np.full(

        (
            title_height,

            comparison_block.shape[1],

            3
        ),

        255,

        dtype=np.uint8
    )


    cv2.putText(

        title_area,

        f"Original file: {image_file.name}",

        (20, 45),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (0, 0, 0),

        2,

        cv2.LINE_AA
    )


    comparison_block = np.vstack(
        (
            title_area,
            comparison_block
        )
    )


    # =================================================================
    # SAVE RESULT
    # =================================================================


    output_filename = (

        image_file.stem

        + "_filter_comparison.jpg"
    )


    output_path = (

        OUTPUT_FOLDER

        / output_filename
    )


    success = cv2.imwrite(

        str(output_path),

        comparison_block
    )


    if success:

        print(
            "Saved:"
        )

        print(
            output_path
        )

    else:

        print(
            "ERROR: Could not save result."
        )


# =====================================================================
# FINISHED
# =====================================================================


print()
print("============================================================")
print("FINISHED")
print("============================================================")
print()

print(
    "Results saved to:"
)

print(
    OUTPUT_FOLDER
)