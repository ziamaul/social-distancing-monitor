import cv2
import numpy as np

def dst2(x1, y1, x2, y2):
    xd = x2 - x1
    yd = y2 - y1

    return xd * xd + yd * yd


# Convert a numpy array or opencv image into something
# that can be used by dearPyGui
def convert_image(image):
    data = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # Convert to RGB in case it's BGR
    data = data.ravel()  # Flatten into 1D array
    data = np.asfarray(data, dtype='f')  # Convert data into 32bit floats

    return np.true_divide(data, 255.0)  # Return normalized image data