import cv2 as cv
import numpy as np
import pickle
import os
import sys

def loadImages(folder):
    images = []
    fileNames = []
    for file in os.listdir(folder):
        img=cv.imread(os.path.join(folder, file))
        img = cv.resize(img,(32,32))
        imgAr = np.asarray(img[:,:])

        red = []
        green = []
        blue = []
        for i in imgAr:
            for j in i:
                red.append(j[0])
                green.append(j[1])
                blue.append(j[2])
        red = np.asarray(red)
        blue = np.asarray(blue)
        green = np.asarray(green)
        imgAr2 = np.concatenate((red,green,blue))
        if imgAr is not None:
            images.append(imgAr)
            fileNames.append(file)
    return images, fileNames

true = loadImages("Aruco/True/")
trueImages = true[0]
trueFileNames = true[1]

false = loadImages("Aruco/False/")
falseImages = false[0]
falseFileNames = false[1]

labels = [1] * len(trueImages) + [0] * len(falseImages)
images = np.concatenate((trueImages, falseImages))
fileNames = trueFileNames + falseFileNames
batch = sys.argv[1] #takes a commandline arg for this

dict = {
    b"batch_label": batch,
    b"labels": labels,
    b"data": images,
    b"filenames": fileNames
}

pickle.dump(dict, open("Aruco/data_batch_" + batch, "wb"))
