import wget
import os
import pip

def input_yn(prompt: str, default: bool = None):
    prompt_yn = "[y/n]"
    if default is True:
        prompt_yn = "[Y/n]"
    elif default is False:
        prompt_yn = "[y/N]"

    v = input(prompt + prompt_yn + ":").strip().lower()

    try:
        return ["y", "yes", "n", "no"].index(v) < 2
    except ValueError:
        if default is None:
            print("Invalid Response")
            input_yn(prompt, default)
        else:
            return default

print("Social Distancing Monitor by Zia Maulana Dewanto (@ziamauld)")
print("")
print("Creating directories...")

if not os.path.exists("./reports"):
    os.mkdir("./reports")

if not os.path.exists("./reports"):
    os.mkdir("./reports")

print("Installing dependencies...")

# TODO: Don't use wrapper
pip.main(["install", "-r", "./requirements.txt"])

if input_yn("\nDownload YOLOv3 pre-trained weights and configurations? This is required for the detector to run. (Approx. filesize 274MB) ", False):
    if not os.path.exists("./YOLO"):
        os.mkdir("./YOLO")

    print("\nyolov3.cfg")
    wget.download("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", "./YOLO/yolov3.cfg")

    print("\nyolov3.weights")
    wget.download("https://pjreddie.com/media/files/yolov3.weights", "./YOLO/yolov3.weights")


