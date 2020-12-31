import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from PIL import Image
from os import listdir
from os.path import isfile, join

# ------------------------

# Opencv File --GIVE FULL PATH OF CASCADES

face_detector_path = r"C:\VIVEK\1.PYTHON_DEV\project\1.CLG_PROJECT\CNN\DEEP_LEARNING\Face_Detection\Face_Detection_With_OpenCV_Haarcascades\Haarcascades\haarcascade_frontalface_default.xml"
eye_detector_path = r"C:\VIVEK\1.PYTHON_DEV\project\1.CLG_PROJECT\CNN\DEEP_LEARNING\Face_Detection\Face_Detection_With_OpenCV_Haarcascades\Haarcascades\haarcascade_eye.xml"
face_detector = cv2.CascadeClassifier(face_detector_path)
eye_detector = cv2.CascadeClassifier(eye_detector_path)


# ------------------------


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def detectFace(img):
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    if len(faces) > 0:
        face = faces[0]
        face_x, face_y, face_w, face_h = face
        img = img[
            int(face_y) : int(face_y + face_h), int(face_x) : int(face_x + face_w)
        ]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img, img_gray
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, img_gray
        # raise ValueError("No face found in the passed image ")


def alignFace(img_path):
    img = cv2.imread(img_path)
    img_raw = img.copy()

    img, gray_img = detectFace(img)

    eyes = eye_detector.detectMultiScale(gray_img, 1.1, 3)
    if len(eyes) >= 2:
        # find the largest 2 eye
        base_eyes = eyes[:, 2]

        items = []
        for i in range(0, len(base_eyes)):
            item = (base_eyes[i], i)
            items.append(item)

        df = pd.DataFrame(items, columns=["length", "idx"]).sort_values(
            by=["length"], ascending=False
        )

        eyes = eyes[df.idx.values[0:2]]

        # --------------------
        # decide left and right eye
        eye_1 = eyes[0]
        eye_2 = eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        # --------------------
        # center of eyes
        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)),
            int(left_eye[1] + (left_eye[3] / 2)),
        )
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]

        right_eye_center = (
            int(right_eye[0] + (right_eye[2] / 2)),
            int(right_eye[1] + (right_eye[3] / 2)),
        )
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]

        cv2.circle(img, left_eye_center, 2, (255, 0, 0), 2)
        cv2.circle(img, right_eye_center, 2, (255, 0, 0), 2)

        # ----------------------
        # find rotation direction
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock

        # ----------------------
        cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
        cv2.line(img, right_eye_center, left_eye_center, (67, 67, 67), 1)
        cv2.line(img, left_eye_center, point_3rd, (67, 67, 67), 1)
        cv2.line(img, right_eye_center, point_3rd, (67, 67, 67), 1)

        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, point_3rd)
        c = euclidean_distance(right_eye_center, left_eye_center)

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi
        if direction == -1:
            angle = 90 - angle

        # --------------------
        # rotate image
        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))

        return new_img
    else:
        print("Problem in eye detected")
        return None
