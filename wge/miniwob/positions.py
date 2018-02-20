"""Module for mapping pixel coordinates to grid points and vice versa."""

import logging


IMAGE_ROWS = 160
IMAGE_COLS = 160
ROW_OFFSET = 50
OUTPUT_ROWS = 20
OUTPUT_COLS = 20


def pixel_coordinates(i, j):
    """Returns the (left, top) pixel coordinates corresponding to the
    (i, j)th action. There is much confusion due to the switching between
    (left, top) coordinate systems and (row, col) coordinate systems.

     ____________
    | . | . | . | <-- actions are in the middle of a hit box
    |___|___|___|
    | . | . | . |
    |___|___|___|

    Args:
        i, j: Int

    Returns:
        (int, int): (left, top)
    """
    output_cols = float(OUTPUT_COLS)
    output_rows = float(OUTPUT_ROWS)
    image_cols = float(IMAGE_COLS)
    image_rows = float(IMAGE_ROWS)
    left = int(
        j * image_cols / output_cols + image_cols / (2 * output_cols))
    top = int(
        i * image_rows / output_rows + image_rows / (2 * output_rows))
    top += ROW_OFFSET
    return left, top


# TODO: Handle cases where bucketing is not quite right
def grid_points(left, top, round_left=None, round_top=None):
    """Give the (left, top) pixel coordinates, returns the (i, j)
    corresponding action pair. Inverse of pixel_coordinates.

    Args:
        left, top: Int
        round_left, round_top = "up" or "down" or None. Specifies
            which direction to round in.

    Returns:
        (Int, Int): (i, j)
    """
    def round(point, diff, direction, minimum, maximum):
        assert direction is None or direction == "up" or \
               direction == "down"
        if diff > 0 and direction == "down":
            return max(point - 1, minimum)
        elif diff < 0 and direction == "up":
            return min(point + 1, maximum)
        else:
            return point

    top -= ROW_OFFSET

    if (not (0 <= left <= IMAGE_COLS) or not (0 <= top <= IMAGE_ROWS)):
        logging.warn("({}, {}) is out of bounds".format(left, top))

    output_cols = float(OUTPUT_COLS)
    output_rows = float(OUTPUT_ROWS)
    image_cols = float(IMAGE_COLS)
    image_rows = float(IMAGE_ROWS)

    # Handle left = image_cols and top = image_rows
    i = int(min(top / (image_rows / output_rows), output_rows - 1))
    j = int(min(left / (image_cols / output_cols), output_cols - 1))

    top += ROW_OFFSET
    grid_left, grid_top = pixel_coordinates(i, j)
    i = round(i, grid_top - top, round_top, 0, int(output_rows) - 1)
    j = round(j, grid_left - left, round_left, 0, int(output_cols) - 1)

    return i, j
