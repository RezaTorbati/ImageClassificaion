import cv2 as cv
import numpy as np
import os
import sys
import pandas as pd


def loadImages(folder):
    files = os.listdir(folder)
    images = np.zeros((len(files), 1024))

    for i in range(0,len(files)):
        img=cv.imread(os.path.join(folder, files[i]))
        img = cv.resize(img,(32,32))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgAr = np.asarray(img[:,:])
        images[i] = imgAr.flatten()

    return images

directory = "../Aruco"

true = pd.DataFrame(data = loadImages(directory + "/True/"))
false = pd.DataFrame(data = loadImages(directory + "/False/"))

true.to_csv(directory + "/True.csv", index = False)
false.to_csv(directory + "/False.csv", index = False)