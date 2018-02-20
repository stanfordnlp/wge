import math


def distance(x1, y1, x2, y2):
    """Regular Euclidean distance"""
    return math.hypot(x2 - x1, y2 - y1)


def rectangle_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    """Computes distance between two rectangles specified by corners.

    Args:
        x1, y1 (float, float): coords of top left corner
        x1b, y1b (float, float): coords of bottom right corner
        x2, y2 (float, float): coords of top left corner
        x2b, y2b (float, float): coords of bottom right corner

    Returns:
        float
    """
    # From https://stackoverflow.com/questions/4978323/
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return distance(x1, y1b, x2b, y2)
    elif left and bottom:
        return distance(x1, y1, x2b, y2b)
    elif bottom and right:
        return distance(x1b, y1, x2, y2b)
    elif right and top:
        return distance(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0.


# TODO (evan): Deprecate this
def row_col_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    """Computes distance between two rectangles specified by corners. The
    distance metric is their x distance if they are in the same row, y
    distance if same col, otherwise inf.

    Args:
        x1, y1 (float, float): coords of top left corner
        x1b, y1b (float, float): coords of bottom right corner
        x2, y2 (float, float): coords of top left corner
        x2b, y2b (float, float): coords of bottom right corner

    Returns:
        float
    """
    def flat_dist(z1, z1b, z2, z2b):
        """z1, ..., z2b, are all x coords or all y coords of a corner.
        Returns the distance between 1 and 2.

        z1 >= z1b, z2 >= z2b"""
        assert z1 >= z1b
        assert z2 >= z2b
        if z1 > z2 and z1b <= z2:
            return 0
        elif z1 > z2:
            return z1b - z2
        elif z2 > z1 and z2b >= z2:
            return 0
        else:
            return z2b - z1

    x_dist = line_segment_distance(x1, x1b, x2, x2b)
    y_dist = line_segment_distance(y1, y1b, y2, y2b) * 3
    if x_dist > 0 and y_dist > 0:
        return float("inf")
    return max(x_dist, y_dist)


def line_segment_distance(start1, end1, start2, end2):
    """Returns the distance between two line segments on the real line.
    Line segments defined by (start1, end1) and (start2, end2) with:

        start1 <= end1, start2 <= end2

    Args:
        start1, end1 (float): start and end of first line segment
        start2, end2 (float): start and end of second line segment

    Returns:
        distance (float)
    """
    assert end1 >= start1
    assert end2 >= start2
    if start1 <= start2 <= end1:
        return 0
    elif start1 <= start2:
        return start2 - end1
    elif start2 <= start1 <= end2:
        return 0
    else:
        return start1 - end2
