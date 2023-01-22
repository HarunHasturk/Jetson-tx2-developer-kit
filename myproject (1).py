import sys
import time
from sys import platform
import numpy as np
import cv2 as cv
import argparse as arg
import os
from flask import Flask, jsonify, request
from openpose import pyopenpose as op
import json

# Kamera açmak için (çalışmayabilir pdf sayfa 85 kontrol et)
cap = cv.VideoCapture(0)


params = dict()
params["model_folder"] = "../../../models/"
params["net_resolution"] = "128x128"
params["number_people_max"] = "1"
params["process_real_time"] = "false"
params["scale_number"] = "1"
params["scale_gap"] = "0.3"
params["disable_multi_thread"] = "True"

opWrapper = op.WrapperPython()

app = Flask(__name__)  # Flask kütüphanesi Merve'nin bahsettiği yer


@app.route('/start', methods=['GET'])
def openpose_exe():
    opWrapper.configure(params)
    opWrapper.start()
    return " OpenPose started succesfully !!"


@app.route('/Coordinates', methods=['GET'])
def get_coo():
    start = time.time()
    ret, frame = cap.read()

    if ret == True:   # Frame görmek için
        cv.imshow("a", frame)
        cv.waitKey(1)

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    print((datum.poseKeypoints).shape)

    coo = getCoordinates(datum.poseKeypoints)
    print(coo)
    print(frame.shape)

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " +
          str(end - start) + " seconds")

    return jsonify(str(coo) + "," + str(frame.shape))

#


def drawLine(arr, img):
    """ Keypoint Locations
    //     {0,  "Nose"},      // Burun
    //     {1,  "Neck"},      // Boyun
    //     {2,  "RShoulder"}, // Omuz
    //     {3,  "RElbow"},    // Dirsek
    //     {4,  "RWrist"},    // Bilek
    //     {5,  "LShoulder"},
    //     {6,  "LElbow"},
    //     {7,  "LWrist"},
    //     {8,  "MidHip"},    // Kalça
    //     {9,  "RHip"},
    //     {10, "RKnee"},
    //     {11, "RAnkle"},    // Ayak Bileği
    //     {12, "LHip"},
    //     {13, "LKnee"},
    //     {14, "LAnkle"},
    //     {15, "REye"},
    //     {16, "LEye"},
    //     {17, "REar"},
    //     {18, "LEar"},
    //     {19, "LBigToe"},
    //     {20, "LSmallToe"},
    //     {21, "LHeel"},
    //     {22, "RBigToe"},
    //     {23, "RSmallToe"},
    //     {24, "RHeel"},       // Topuk
    //     {25, "Background"}
    // };

    """
    blank_image = np.zeros(shape=img.shape, dtype=np.uint8)

    cv.line(blank_image, (arr[0][1][0], arr[0][1][1]),
            (arr[0][8][0], arr[0][8][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][1][0], arr[0][1][1]),
            (arr[0][2][0], arr[0][2][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][1][0], arr[0][1][1]),
            (arr[0][5][0], arr[0][5][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][2][0], arr[0][2][1]),
            (arr[0][3][0], arr[0][3][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][3][0], arr[0][3][1]),
            (arr[0][4][0], arr[0][4][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][5][0], arr[0][5][1]),
            (arr[0][6][0], arr[0][6][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][6][0], arr[0][6][1]),
            (arr[0][7][0], arr[0][7][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][8][0], arr[0][8][1]),
            (arr[0][9][0], arr[0][9][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][9][0], arr[0][9][1]),
            (arr[0][10][0], arr[0][10][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][10][0], arr[0][10][1]),
            (arr[0][11][0], arr[0][11][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][8][0], arr[0][8][1]),
            (arr[0][12][0], arr[0][12][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][12][0], arr[0][12][1]),
            (arr[0][13][0], arr[0][13][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][13][0], arr[0][13][1]),
            (arr[0][14][0], arr[0][14][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][1][0], arr[0][1][1]),
            (arr[0][0][0], arr[0][0][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][0][0], arr[0][0][1]),
            (arr[0][15][0], arr[0][15][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][15][0], arr[0][15][1]),
            (arr[0][17][0], arr[0][17][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][0][0], arr[0][0][1]),
            (arr[0][16][0], arr[0][16][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][16][0], arr[0][16][1]),
            (arr[0][18][0], arr[0][18][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][14][0], arr[0][14][1]),
            (arr[0][19][0], arr[0][19][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][19][0], arr[0][19][1]),
            (arr[0][20][0], arr[0][20][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][14][0], arr[0][14][1]),
            (arr[0][21][0], arr[0][21][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][11][0], arr[0][11][1]),
            (arr[0][22][0], arr[0][22][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][22][0], arr[0][22][1]),
            (arr[0][23][0], arr[0][23][1]), (255, 255, 255), 3)
    cv.line(blank_image, (arr[0][11][0], arr[0][11][1]),
            (arr[0][24][0], arr[0][24][1]), (255, 255, 255), 3)

    overlay = blank_image.copy()

    for i in range(0, 24):

        if i == 19 or i == 20 or i == 22 or i == 23:
            continue
        else:
            cv.circle(overlay, (arr[0][i][0], arr[0]
                      [i][1]), 10, (0, 255, 255), -1)

    opacity = 0.7
    cv.addWeighted(overlay, opacity, blank_image, 1 - opacity, 0, blank_image)

    return blank_image


# Pozları geribildirim olarak atıyor stackoverflow'daki kod çalıştı değiştirdim
def getCoordinates(arr):

    crd = ""

    for i in range(0, 25):
        if arr.shape != (1, 25, 3):
            for i in range(0, 25):
                if i == 24:
                    crd += str(0) + "," + str(0)
                else:
                    crd += str(0) + "," + str(0) + ","
            return crd

        elif i == 24:
            crd += str(arr[0][i][0]) + "," + str(arr[0][i][1])
        else:
            crd += str(arr[0][i][0]) + "," + str(arr[0][i][1]) + ","

    return crd


def start_server():  # flask serveri başlatıyoruz buradan
    app.run(debug=False, host="10.1.22.28", port=5050)


if __name__ == "__main__":

    try:

        start_server()

    except Exception as ex:
        print(ex)

cap.release()  # Turn off the camera
sys.exit(-1)
