import cv2 as cv
import numpy as np
import configparser
import threading
import time
import math
import queue
import os

from src.utilities import transform, utils
from src import gui
net: cv.dnn
input_dim = None

pixels_to_meters: int
matrix: np.matrix

cap: cv.VideoCapture
detection_thread: threading.Thread
min_conf = 0


def init(weights_path, config_path):
    print("[DETECTION] Initialized")
    global net
    net = cv.dnn.readNet(weights_path, config_path)

    parser = configparser.ConfigParser(strict=False)
    parser.read(config_path)

    global input_dim
    input_dim = (parser.getint("net", "width"), parser.getint("net", "height"))


def start(gui_queue: queue.Queue, status_queue: queue.Queue, data_queue: queue.Queue, image_queue: queue.Queue, message_queue: queue.Queue, settings: dict):
    status_queue.put(["STARTING", "STARTING"])

    errored = False

    try:
        init(settings["model_weights_path"], settings["model_config_path"])
    except:
        message_queue.put({"type": "error", "title": "ERROR LOADING NETWORK", "message": ".cfg and .weights files not found. Change this in the settings and make sure the files exist, then restart the detector."})
        errored = True

    global cap
    cap = cv.VideoCapture(settings["camera_index"], cv.CAP_DSHOW)
    cap.set(propId=cv.CAP_PROP_FRAME_WIDTH, value=settings["camera_width"])
    cap.set(propId=cv.CAP_PROP_FRAME_HEIGHT, value=settings["camera_height"])

    if not cap.isOpened():
        message_queue.put({"type": "error", "title": "ERROR OPENING CAMERA", "message": "The input device at USB port {0} cannot be opened. Make sure it is connected or change the port in the settings, then restart the detector.".format(settings["camera_index"])})
        errored = True

    status_queue.put(["STARTING", "WORKING"])

    try:
        global matrix
        global pixels_to_meters
        matrix, pixels_to_meters = calibrate(cap.read()[1], settings["calib_data_path"], message_queue)
    except:
        errored = True

    if errored:
        status_queue.put(["ERROR", "ERROR"])

        # Fallback to restart network
        while True:
            if not gui_queue.empty():
                _ = gui_queue.get()
                if not gui_queue.empty() and gui_queue.get():
                    start(gui_queue, status_queue, data_queue, image_queue, message_queue, settings)

                return

    running = True
    status_queue.put(["OK", "OK"])
    while running:
        _, img = cap.read()

        frametime = time.time()
        detection_data, images = detect(img)
        frametime = int((time.time() - frametime) * 1000)
        detection_data.append(frametime)

        #print("[DETECTION] took " + str(frametime) + "ms")

        data_queue.put(detection_data)
        image_queue.put([
            utils.convert_image(cv.resize(images[0], (1080, 720))),
            utils.convert_image(cv.resize(images[1], (1080, 720)))
        ])

        if not gui_queue.empty():
            running = gui_queue.get()

    print("[DETECTION] Stopped")

    # pass in TRUE again to restart
    if not gui_queue.empty() and gui_queue.get():
        status_queue.put(["PAUSED", "PAUSED"])
        data_queue.put([-1, -1, -1, -1, -1, -1])
        image_queue.put([np.zeros((1080, 720, 4), dtype=np.float32)] * 2)
        return start(gui_queue, status_queue, data_queue, image_queue, message_queue, settings)

def detect(img):
    #print("[DETECTION] Detecting")
    height, width, _ = img.shape

    # Preparing the image
    blob = cv.resize(img.copy(), input_dim)
    blob = cv.dnn.blobFromImage(blob, 1/input_dim[0], input_dim, (0, 0, 0), swapRB=True, crop=False)

    bev = cv.warpPerspective(img, matrix, (width, height))

    # Detecting

    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # NMS and output
    boxes, confidences, indexes, bottoms, classes, projected = parse(outputs, width, height, matrix)

    avg_dist = -1
    unsure_count = 0
    total_count = 0
    violations = 0
    confirmed_count = 0
    if len(indexes) > 0:
        indexes_flat = indexes.flatten()
        for i in indexes_flat:
            x, y, w, h = boxes[i]
            bx, by = bottoms[i]
            px, py = projected[i]

            total_count += 1
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1, 1)
            if classes[i] != 0:
                unsure_count += 1
                cv.circle(bev, (px, py), 2, (0, 200, 255))
                cv.circle(img, (bx, by), 2, (0, 200, 255))
                continue

            min_dist = math.inf

            for other in indexes.flat:
                if other == i or classes[i] != 0:
                    continue
                opx, opy = projected[other]

                dst = utils.dst2(px, py, opx, opy)
                if dst < min_dist:
                    min_dist = dst

            avg_dist += min_dist
            if min_dist / pixels_to_meters < 2:
                cv.circle(bev, (px, py), 4, (0, 0, 255), -1)
                cv.circle(img, (bx, by), 4, (0, 0, 255), -1)
                violations += 1
            else:
                cv.circle(bev, (px, py), 4, (0, 255, 0), -1)
                cv.circle(img, (bx, by), 4, (0, 255, 0), -1)

        confirmed_count = total_count - unsure_count
        if confirmed_count < 2:
            avg_dist = -1
        else:
            avg_dist = avg_dist / confirmed_count

    return [
        total_count,
        confirmed_count,
        unsure_count,
        avg_dist / pixels_to_meters,
        violations
    ], [
        img,
        bev,
    ]

# UTILITIES


def parse(outputs, img_width, img_height, proj_matrix):
    boxes = []
    confidences = []
    bottom = []
    classes = []
    projected = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)

            confidence = scores[class_id]

            if confidence > min_conf:
                # Center of the box
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)

                # Box dimensions
                w = int(detection[2] * img_width)
                h = int(detection[3] * img_height)

                # Top left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bx = int(center_x)
                by = int(center_y + h / 2)

                prjx, prjy = transform.project(proj_matrix, bx, by)
                projected.append((int(prjx), int(prjy)))

                # The bottom of the box, usually where the detected person's feet touches the floor.
                bottom.append([bx, by])

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

                classes.append(class_id)

    # Apply Non-Maxima Supression to remove overlapping boxes
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return boxes, confidences, indexes, bottom, classes, projected


def calibrate(img, calib_file: str, message_queue: queue.Queue):
    mat: np.matrix
    p2m: int

    if os.path.exists(calib_file):
        with open(calib_file, "r") as data:
            try:
                d1, d2 = data.readlines()
                p2m = int(d2)
                mat = np.matrix(d1, dtype=np.float32)
                data.close()
            except:
                message_queue.put({"type": "error", "title": "ERROR CALIBRATING", "message": "Calibration data file malformed. Delete the file then restart the detector to recalibrate."})

    else:
        print("[DETECTION] Detecting markers")
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
        aruco_params = cv.aruco.DetectorParameters_create()

        corners, ids, rejected = cv.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

        if len(ids) != 4:
            message_queue.put({"type": "error", "title": "ERROR CALIBRATING", "message": "{0} out of 4 calibration markers cannot be detected. Readjust the markers or the camera then restart the detector to recalibrate.".format(4 - len(ids))})
            return None

        print("[DETECTION] Calculating transformation matrix")
        markers = []
        for i in range(0, 4):
            marker_id = ids[i]
            markers[marker_id - 1] = corners[i]

        mat, _ = transform.get_transform_matrices(
            [
                markers[0][0],
                markers[1][1],
                markers[2][3],
                markers[3][2]
            ],
            (
                img.shape[1],
                img.shape[0]
            )
        )[0]

        # Obtain real-world distances
        # Assume markers 1 and 2 are approximately 4 meters apart.
        m1x, m1y = transform.project(mat, markers[0][0][0], markers[0][0][1])
        m2x, m2y = transform.project(mat, markers[1][1][0], markers[1][1][1])

        p2m = utils.dst2(m1x, m1y, m2x, m2y) / 4

        print("[DETECTION] Writing calibration data")
        with open(calib_file, "w") as file:
            data = ""
            for i in range(1, 10):
                data += str(mat.item(i - 1)) + " "
                if i % 3 == 0 and i + 1 != 10:
                    data += "; "

            file.write(data)
            file.write("\n" + str(pixels_to_meters))
            file.close()


    print("\n[DETECTION] Calibration finished:")
    print(p2m)
    print(mat)
    print("\n")
    return mat, p2m
