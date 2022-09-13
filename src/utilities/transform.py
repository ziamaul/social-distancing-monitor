import numpy as np
import cv2 as cv


def extend_line(p1, p2, height):
    x1, y1 = p1
    x2, y2 = p2

    print(p1)
    print(p2)
    gradient = (y1 - y2) / (x2 - x1)

    print(gradient)

    top_x = x1 + (y1 / gradient)
    bot_x = x1 - ((height - y1) / gradient)

    return (top_x, 0), (bot_x, height)


# Generate the transformation matrix and its inverse from marker points.
# [marker1, marker2, marker3, marker4]
def get_transform_matrices(points, dim):
    width, height = dim

    m1 = points[0]
    m2 = points[1]
    m3 = points[2]
    m4 = points[3]

    l1_top, l1_bot = extend_line(m1, m3, height)
    l2_top, l2_bot = extend_line(m2, m4, height)

    src = np.float32([l1_bot, l2_bot, [0, 0], [width, 0]])
    dst = np.float32([[l1_top[0], l1_bot[1]], [l2_top[0], l2_bot[1]], [0, 0], [width, 0]])

    print(src)
    print(dst)

    return cv.getPerspectiveTransform(src, dst), cv.getPerspectiveTransform(dst, src)


# Projects a point
def project(matrix, x, y):
    x = ((matrix.item((0, 0)) * x) + (matrix.item((0, 1)) * y) + (matrix.item((0, 2)))) / ((matrix.item((2, 0)) * x) + (matrix.item((2, 1)) * y) + (matrix.item((2, 2))))
    y = ((matrix.item((1, 0)) * x) + (matrix.item((1, 1)) * y) + (matrix.item((1, 2)))) / ((matrix.item((2, 0)) * x) + (matrix.item((2, 1)) * y) + (matrix.item((2, 2))))

    return x, y