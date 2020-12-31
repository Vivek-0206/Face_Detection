"""
    A function to detect face from directory and store into new directory
"""
import cv2
import os
from os import listdir
from os.path import isfile, join

# function to detect face using OpenCV
def detect_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # load OpenCV face detector, I am using LBP which is fast
    face_cascade = cv2.CascadeClassifier("Haarcascades/lbpcascade_frontalface.xml")
    # Detect multiscale images
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces


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
        image = cv2.imread(image_path)
        faces = detect_face(image)
        for (x, y, w, h) in faces:
            face = image[y : y + h, x : x + w]
        path = (
            des
            + "face_"
            + image_names[i].split(".")[0]
            + str(image_names[i].split(".")[1])
            + ".jpg"
        )
        cv2.imwrite(path, face)
        print("Done image:", image_names[i])
