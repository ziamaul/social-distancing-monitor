import time
import dearpygui.dearpygui as dpg
import queue
import os

# Currently Windows-only
if os.name == "nt":
    from pygrabber.dshow_graph import FilterGraph
    filter_graph = FilterGraph()

cameras = []

graph_times = []
for i in range(0, 60):
    graph_times.append(i)

frame_times = [0] * 60
avg_distances = [0] * 60
confirmed_counts = [0] * 60
unsure_counts = [0] * 60
total_counts = [0] * 60
violation_counts = [0] * 60

last_path = "./reports"

max_total_counts = 5
total_axis, confirmed_axis, unsure_axis = None, None, None

graphs = []


def add_graph(title, tag, x_axis, y_axis, y_label, min_value, max_value):
    with dpg.plot(label=title, width=720, height=200):
        dpg.add_plot_axis(dpg.mvXAxis, show=False, tag=str(tag + "_x"), no_gridlines=True)
        y = dpg.add_plot_axis(dpg.mvYAxis, label=y_label, tag=str(tag + "_y"))
        dpg.add_line_series(graph_times, frame_times, parent=str(tag + "_y"), tag=tag)

    graphs.append([tag, x_axis, y_axis, y, min_value, max_value])


def update_graphs():
    for graph in graphs:
        dpg.set_value(graph[0], [graph[1], graph[2]])
        graph[5] = max(graph[2][0], graph[5])
        dpg.set_axis_limits(graph[3], graph[4], graph[5])


def show_file_selector(item_name, callback):
    dpg.set_item_callback(item_name, callback)
    dpg.show_item(item_name)


def scan_cameras():
    if os.name == "nt":
        global cameras
        cameras = filter_graph.get_input_devices()


def init(gui_queue: queue.Queue, status_queue: queue.Queue, data_queue: queue.Queue, image_queue: queue.Queue, reporter_queue: queue.Queue, reporter_data: queue.Queue, settings: dict):
    print("[GUI      ] Initialized")

    scan_cameras()

    dpg.create_context()
    dpg.create_viewport(title='Social Distancing Monitor', width=1030, height=720, resizable=False, max_width=1030, max_height=720)

    def reset(sender, app_data, user_data):
        print("[APP] RESET")
        if os.name == "nt":
            dpg.set_value("camera_stream", "Streaming From: " + cameras[settings["camera_index"]])

        dpg.hide_item("reset_button")
        user_data[1].put(last_path)
        user_data[1].put(True)
        user_data[1].put(True)

        user_data[0].put(False)
        user_data[0].put(True)

    with dpg.texture_registry():
        width, height, channels, data = dpg.load_image("./src/utilities/no_input.png")
        dpg.add_dynamic_texture(width=1080, height=720, default_value=data, tag="bev_image")
        dpg.add_dynamic_texture(width=1080, height=720, default_value=data, tag="output_image")

    with dpg.window(tag="settings_window", label="Settings", no_resize=True, width=480, height=250, pos=(275, 235), show=False, no_move=True, no_collapse=True, modal=True):
        with dpg.table(header_row=False):

            dpg.add_table_column(width_fixed=True, width=100)
            dpg.add_table_column()
            dpg.add_table_column(width_fixed=True, width=50)

            with dpg.table_row():
                dpg.add_text("Model Weights File")
                dpg.add_input_text(default_value=settings["model_weights_path"], width=-1, readonly=True, tag="model_weights")

                def set_model_weights(sender, app_data, user_data):
                    settings["model_weights_path"] = app_data["file_path_name"]
                    dpg.set_value("model_weights", app_data["file_path_name"])

                dpg.add_button(label="Set", callback=lambda: show_file_selector("weights_selector", set_model_weights))

            with dpg.table_row():
                dpg.add_text("Model Config File")
                dpg.add_input_text(default_value=settings["model_config_path"], width=-1, readonly=True, tag="model_config")

                def set_model_configs(sender, app_data, user_data):
                    settings["model_config_path"] = app_data["file_path_name"]
                    dpg.set_value("model_config", app_data["file_path_name"])

                dpg.add_button(label="Set", callback=lambda: show_file_selector("cfg_selector", set_model_configs))

            with dpg.table_row():
                dpg.add_text("Reports Path")
                dpg.add_input_text(default_value=settings["reports_path"], width=-1, readonly=True, tag="reports_path")
                def set_reports_path(sender, app_data, user_data):
                    settings["reports_path"] = app_data["file_path_name"]
                    dpg.set_value("reports_path", app_data["file_path_name"])

                    reporter_queue.put(app_data["file_path_name"])
                    reporter_queue.put(True)

                dpg.add_button(label="Set", callback=lambda: show_file_selector("path_selector", set_reports_path))

            with dpg.table_row():
                dpg.add_spacer(height=10)

            with dpg.table_row():
                dpg.add_text("Camera Port")
                with dpg.group():
                    def set_index(sender, app_data, user_data):
                        settings["camera_index"] = app_data
                        if os.name == "nt":
                            dpg.set_value("camera_index_combo", cameras[app_data])

                    dpg.add_input_int(default_value=settings["camera_index"], width=-1, tag="camera_index", on_enter=True, callback=set_index)

                    if os.name == "nt":
                        def set_index_combo(sender, app_data, user_data):
                            index = cameras.index(app_data)
                            settings["camera_index"] = index
                            dpg.set_value("camera_index", index)

                        dpg.add_combo(default_value=cameras[settings["camera_index"]], items=cameras, callback=set_index_combo, tag="camera_index_combo", width=-1)
            with dpg.table_row():
                dpg.add_text("Frame Width")
                def set_width(sender, app_data, user_data):
                    settings["camera_width"] = app_data

                dpg.add_input_int(default_value=settings["camera_width"], width=-1, tag="camera_width", on_enter=True, callback=set_width)


            with dpg.table_row():
                dpg.add_text("Frame Height")

                def set_height(sender, app_data, user_data):
                    settings["camera_height"] = app_data

                dpg.add_input_int(default_value=settings["camera_height"], width=-1, tag="camera_height", on_enter=True, callback=set_height)

        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply", width=100, callback=reset, tag="menu_apply")
            dpg.set_item_user_data("menu_apply", [gui_queue, reporter_queue])
            with dpg.tooltip(parent="menu_apply"):
                dpg.add_text("Apply changes. Will reset the detector.")


            if os.name == "nt":
                def scan_cameras_update():
                    scan_cameras()
                    dpg.configure_item("camera_index_combo", items=cameras)

                dpg.add_button(label="Scan Inputs", width=100, callback=scan_cameras_update, tag="input_scan")
                with dpg.tooltip(parent="input_scan"):
                    dpg.add_text("Scans USB ports for external cameras.")

    with dpg.window(tag="main_window", no_resize=True):

        dpg.add_file_dialog(label="Select Directory", modal=True, directory_selector=True, show=False, tag="path_selector", width=880, height=520)

        with dpg.file_dialog(label="Select File", modal=True, show=False, tag="weights_selector", width=880, height=520):
            dpg.add_file_extension(".weights")

        with dpg.file_dialog(label="Select File", modal=True, show=False, tag="cfg_selector", width=880, height=520):
            dpg.add_file_extension(".cfg")

        with dpg.group(tag="main_group", horizontal=True):
            with dpg.window(tag="main_info", width=265, height=720, no_scrollbar=True, no_move=True, no_resize=True, no_collapse=True, no_close=True, no_title_bar=True):
                dpg.add_text(default_value="Time: XX:XX:XX", tag="time_label")

                dpg.add_text("Process Status")
                with dpg.table(tag="info_process", header_row=False, width=250, borders_outerH=True,
                               borders_outerV=True):
                    dpg.add_table_column()
                    dpg.add_table_column()

                    with dpg.table_row():
                        dpg.add_text("Frame Time")
                        dpg.add_text("-", tag="frametime_status")

                    with dpg.table_row():
                        dpg.add_text("Calibration")
                        dpg.add_text("-", tag="calibration_status")

                    with dpg.table_row():
                        dpg.add_text("Detection")
                        dpg.add_text("-", tag="detection_status")

                dpg.add_spacer(height=20)
                dpg.add_text("Detection Information")
                with dpg.table(tag="info_detection", header_row=False, width=250, borders_outerH=True,
                               borders_outerV=True):
                    dpg.add_table_column()
                    dpg.add_table_column()

                    with dpg.table_row():
                        dpg.add_text("Total")
                        dpg.add_text("-", tag="total_count")

                    with dpg.table_row():
                        dpg.add_text("Confirmed")
                        dpg.add_text("-", tag="confirmed_count")

                    with dpg.table_row():
                        dpg.add_text("Unsure")
                        dpg.add_text("-", tag="unsure_count")

                with dpg.table(tag="info_statistics", header_row=False, width=250, borders_outerH=True,
                               borders_outerV=True):
                    dpg.add_table_column()
                    dpg.add_table_column()

                    with dpg.table_row():
                        dpg.add_text("Average Distance")
                        dpg.add_text("-", tag="avg_distance_count")

                    with dpg.table_row():
                        dpg.add_text("Violations")
                        dpg.add_text("-", tag="violations_count")

                dpg.add_spacer(height=20)

                # FUNCTIONS
                dpg.add_button(label="Settings", width=250, callback=lambda: dpg.show_item("settings_window"))

                dpg.add_button(label="Reset Detector", width=250, callback=reset, tag="reset_button")
                dpg.set_item_user_data("reset_button", [gui_queue, reporter_queue])

            with dpg.window(tag="visuals_bar", width=750, pos=(265, 0), no_move=True, no_resize=True, no_collapse=True, no_close=True, no_title_bar=True):
                with dpg.tab_bar(tag="tabs"):
                    with dpg.tab(label="Camera"):
                        if os.name == "nt":
                            dpg.add_text("Streaming From: " + cameras[settings["camera_index"]], tag="camera_stream")
                        dpg.add_image("output_image", width=720, height=480)
                        dpg.add_image("bev_image", width=720, height=480)

                    with dpg.tab(label="Graphs"):

                        with dpg.collapsing_header(label="Performance"):
                            add_graph("Frame Times", "frame_times_graph", graph_times, frame_times, "milliseconds (ms)", 0, 1000)

                            with dpg.plot(label="Detection Accuracy", width=720, height=200):
                                dpg.add_plot_legend()

                                dpg.add_plot_axis(dpg.mvXAxis, show=False, no_gridlines=True)
                                dpg.add_plot_axis(dpg.mvYAxis, tag="total_axis")
                                dpg.set_axis_limits("total_axis", 0, 5)

                                dpg.add_line_series(graph_times, total_counts, parent="total_axis", tag="total_graph", label="Total Detections")
                                dpg.add_line_series(graph_times, confirmed_counts, parent="total_axis", tag="confirmed_graph", label="Confirmed Detections")
                                dpg.add_line_series(graph_times, unsure_counts, parent="total_axis", tag="unsure_graph", label="Unsure Detections")

                            dpg.add_spacer(height=20)

                        with dpg.collapsing_header(label="Crowd Information"):
                            add_graph("Average Distance", "average_distance_graph", graph_times, avg_distances, "meters (m)", 0, 10)
                            add_graph("Violations", "violations_graph", graph_times, violation_counts, "", 0, 10)


    start(gui_queue, status_queue, data_queue, image_queue, reporter_queue, reporter_data, settings)


def start(gui_queue: queue.Queue, status_queue: queue.Queue, data_queue: queue.Queue, image_queue: queue.Queue, reporter_queue: queue.Queue, reporter_data: queue.Queue, settings: dict):
    global max_total_counts
    print("[GUI      ] Started")
    dpg.setup_dearpygui()
    dpg.show_viewport()

    dpg.set_primary_window("main_window", True)

    while dpg.is_dearpygui_running():
        localtime = time.localtime(time.time())
        dpg.set_value("time_label", "Time: {0}:{1}:{2}".format(localtime.tm_hour, localtime.tm_min, localtime.tm_sec))
        update_graphs()

        if not status_queue.empty():
            status = status_queue.get()
            if status[0]=="OK":
                dpg.show_item("reset_button")

            dpg.set_value("calibration_status", status[0])
            dpg.set_value("detection_status", status[1])

        if not data_queue.empty():
            data = data_queue.get()

            total_counts.insert(0, data[0])
            total_counts.pop()
            confirmed_counts.insert(0, data[1])
            confirmed_counts.pop()
            unsure_counts.insert(0, data[2])
            unsure_counts.pop()
            violation_counts.insert(0, data[4])
            violation_counts.pop()

            reporter_data.put([
                data[3],
                data[4],
                data[5],
                data[0],
                data[1],
                data[2]
            ])

            max_total_counts = max(total_counts[0], max_total_counts)
            dpg.set_axis_limits("total_axis", 0, max_total_counts)

            dpg.set_value("total_graph", [graph_times, total_counts])
            dpg.set_value("confirmed_graph", [graph_times, confirmed_counts])
            dpg.set_value("unsure_graph", [graph_times, unsure_counts])

            dpg.set_value("total_count", data[0])
            dpg.set_value("confirmed_count", data[1])
            dpg.set_value("unsure_count", data[2])
            if data[3] == -1:
                dpg.set_value("avg_distance_count", "-")
                avg_distances.insert(0, 0)
            else:
                dpg.set_value("avg_distance_count", "~" + str(round(data[3] * 100) / 100) + "m")
                avg_distances.insert(0, data[3])

            avg_distances.pop()

            dpg.set_value("violations_count", data[4])

            if data[5] == -1:
                dpg.set_value("frametime_status", "-")
                frame_times.insert(0, 0)
            else:
                dpg.set_value("frametime_status", str(data[5]) + "ms")
                frame_times.insert(0, data[5])

            frame_times.pop()

        if not image_queue.empty():
            images = image_queue.get()
            dpg.set_value("output_image", images[0])
            dpg.set_value("bev_image", images[1])

        dpg.render_dearpygui_frame()

    print("[GUI      ] Closed")
    dpg.destroy_context()
