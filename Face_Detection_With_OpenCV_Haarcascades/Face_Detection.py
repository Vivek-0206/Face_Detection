import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from PIL import Image
from os import listdir
from os.path import isfile, join

from Face_Alignment import alignFace, detectFace


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
    for i, image_path in enumerate(image_file_path):
        alignedFace = alignFace(image_path)
        if alignedFace.all() != None:
            img, gray_img = detectFace(alignedFace)
            path = (
                des
                + "face_"
                + image_names[i].split(".")[0]
                + str(image_names[i].split(".")[1])
                + ".jpg"
            )
            cv2.imwrite(path, img)
            print("Done image:", image_names[i])
        else:
            break
