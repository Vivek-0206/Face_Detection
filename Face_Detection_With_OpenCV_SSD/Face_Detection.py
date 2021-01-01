"""
    A function to detect face from directory and store into new directory
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join


def load_model():

    prototxt = r"C:\VIVEK\1.PYTHON_DEV\project\1.CLG_PROJECT\Face_Detection\Face_Detection_With_OpenCV_SSD\Model\deploy.prototxt"
    model = r"C:\VIVEK\1.PYTHON_DEV\project\1.CLG_PROJECT\Face_Detection\Face_Detection_With_OpenCV_SSD\Model\res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)

    return detector


def detect_face(image):
    """
    Detect face from image.
    """
    detector = load_model()
    original_size = image.shape
    target_size = (300, 300)
    image = cv2.resize(image, target_size)  # Resize to target_size
    aspect_ratio_x = original_size[1] / target_size[1]
    aspect_ratio_y = original_size[0] / target_size[0]
    imageBlob = cv2.dnn.blobFromImage(image=image)
    detector.setInput(imageBlob)
    detections = detector.forward()

    return detections, aspect_ratio_x, aspect_ratio_y


def face_detection(src, des):
    """
	Detect image from src folder and store face data into des folder
	"""
    image_file_path = []
    image_names = []
    # Directory of image of persons we'll be extracting faces from
    mypath = src

    # check passed db folder exists
    if os.path.isdir(mypath) == True:
        for r, d, f in os.walk(mypath):  # r=root, d=directories, f = files
            for file in f:
                if ".jpg" in file:
                    # exact_path = os.path.join(r, file)
                    exact_path = r + "/" + file
                    image_file_path.append(exact_path)
                    image_names.append(file)
        print("Image names Collected")

    if len(image_file_path) == 0:
        print(
            f"WARNING: There is no image in this path {mypath}. Face recognition will not be performed."
        )
    for j, image_path in enumerate(image_file_path):
        image = cv2.imread(image_path)
        base_img = image.copy()
        detections, aspect_ratio_x, aspect_ratio_y = detect_face(image)

        detections_df = pd.DataFrame(
            detections[0][0],
            columns=[
                "img_id",
                "is_face",
                "confidence",
                "left",
                "top",
                "right",
                "bottom",
            ],
        )
        detections_df = detections_df[detections_df["is_face"] == 1]
        detections_df = detections_df[detections_df["confidence"] >= 0.93]

        for i, instance in detections_df.iterrows():
            left = int(instance["left"] * 300)
            bottom = int(instance["bottom"] * 300)
            right = int(instance["right"] * 300)
            top = int(instance["top"] * 300)
            confidence_score = str(round(100 * instance["confidence"], 2)) + " %"
            detected_face = base_img[
                int(top * aspect_ratio_y) - 100 : int(bottom * aspect_ratio_y) + 100,
                int(left * aspect_ratio_x) - 100 : int(right * aspect_ratio_x) + 100,
            ]
            path = (
                des
                + "face_"
                + image_names[j].split(".")[0]
                + str(image_names[j].split(".")[1])
                + ".jpg"
            )
            cv2.imwrite(path, detected_face)
            print(
                f"Done image {image_names[j]} with confidence score {confidence_score}"
            )
    print(f"Saved image to group_of_faces")

