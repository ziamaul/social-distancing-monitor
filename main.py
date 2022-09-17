import queue
import threading
import os
import sys

from src import detector, gui, reporter

def main(settings: dict):
    gui_queue = queue.Queue()
    reporter_queue = queue.Queue()

    message_queue = queue.Queue()

    status_queue = queue.Queue()
    data_queue = queue.Queue()
    image_queue = queue.Queue()

    reporter_data = queue.Queue()

    detector_thread = threading.Thread(target=detector.start, name="DETECTION", args=(gui_queue, status_queue, data_queue, image_queue, message_queue, settings))
    reporter_thread = threading.Thread(target=reporter.start, name="REPORTER", args=(reporter_queue, reporter_data, settings))

    print("[DETECTION] Started")
    print("[REPORTER ] Started")
    detector_thread.start()
    reporter_thread.start()

    print("[GUI      ] Started")

    try:
        gui.init(gui_queue, status_queue, data_queue, image_queue, reporter_queue, reporter_data, message_queue, settings)
    finally:
        gui_queue.put(False)
        reporter_queue.put(None)
        reporter_queue.put(False)

    detector_thread.join()
    reporter_thread.join()

    print("[APP      ] Closed")


if __name__ == "__main__":

    # Default Settings
    settings_dict = {
        "model_weights_path": "./YOLO/yolov3.weights",
        "model_config_path": "./YOLO/yolov3.cfg",

        "camera_index": 0,
        "camera_width": 1920,
        "camera_height": 1080,

        "reports_path": "./reports"
    }

    print("[APP      ] Loading settings")

    if os.path.exists("./data/settings"):
        with open("./data/settings", "r") as file:
            setting_entries = file.read().split(";")
            i = 0

            while i + 1 < len(setting_entries):
                key = setting_entries[i]
                value = setting_entries[i+1]

                if value.isdigit():
                    value = int(value)

                settings_dict[key] = value
                i += 2

    main(settings_dict)

    with open("./data/settings", "w") as file:
        settings = ""
        for key, value in settings_dict.items():
            settings += str(key) + ";" + str(value) + ";"
        file.write(settings)
        file.close()
