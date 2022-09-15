import time
import os
import csv
import queue

def start(reporter_queue: queue.Queue, reporter_data: queue.Queue, settings: dict):
    running = True
    last_time = (time.time() % 3600) / 60 % 30

    data = [0, 0, 0, 0, 0, 0, 0]
    data_length = 0

    reports_path = settings["reports_path"]

    while running:
        # Will be in sync with global clock, make report every 30 minutes regardless of start time.
        current_time = (time.time() % 3600) / 60 % 30

        while not reporter_data.empty():
            new_data = reporter_data.get()
            data_length += 1

            for i in range(0, len(new_data)):
                data[i] += new_data[i]

        if current_time < last_time:
            if data_length != 0:
                for i in range(0, len(data)):
                    data[i] /= data_length

            data[5] = data_length

            report(data, reports_path)

            data = [0, 0, 0, 0, 0, 0, 0]

        last_time = current_time

        if not reporter_queue.empty():
            reports_path = reporter_queue.get()
            running = reporter_queue.get()

    if not reporter_queue.empty() and reporter_queue.get():
        start(reporter_queue, reporter_data)

    print("[REPORTER ] Closed")



def report(data: list, path: str):
    localtime = time.localtime(time.time())
    report_name = "Monitor_Report-{0}_{1}_{2}.csv".format(localtime.tm_mday, localtime.tm_mon, localtime.tm_year)
    report_path = path + "/" + report_name

    if not os.path.exists(path):
        os.mkdir(path)

    no_report = not os.path.exists(report_path)

    with open(report_path, "a", newline='') as file:
        writer = csv.writer(file)

        if no_report:
            writer.writerow(["unix_time", "local_time", "avg_dist", "violations", "frame_time", "total_detections", "confirmed_detections", "unsure_detections", "frames_processed"])

        hms = "{0}:{1}:{2}".format(localtime.tm_hour, localtime.tm_min, localtime.tm_sec)

        #print("[REPORTER] Writing report for " + hms)
        data.insert(0, hms)
        data.insert(0, time.time())

        writer.writerow(data)
