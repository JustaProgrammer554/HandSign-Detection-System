import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

OFFSET = 20
IMGSIZE = 300

folder = "EnterFolderName"
counter = 0

while True:
    ret, img = cap.read() # get info from the cam
    hands, img = detector.findHands(img) # get info from the hand positions

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((IMGSIZE, IMGSIZE, 3), np.uint8) * 255 # creating the white background for the data img
        imgCrop = img[y - OFFSET : y + h + OFFSET, x - OFFSET : x + w + OFFSET] # crop the hand for data img

        imgCropShape = imgCrop.shape # sizes of the imgCrop[h, w, channel]

        aspectRatio = h / w # the ratio between the height and the width of the imgCrop

        if aspectRatio > 1: # check if the img is rectangle
            k = IMGSIZE / h
            wCal = math.ceil(k * w) # calculate the width that need to be expanded
            imgResize = cv2.resize(imgCrop, (wCal, IMGSIZE))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((IMGSIZE - wCal)/2) # calculate the gap to center the width of imgCrop in imgWhite
            imgWhite[:, wGap : wCal + wGap] = imgResize  # put the imgCrop on the imgWhite

        else:
            k = IMGSIZE / w
            hCal = math.ceil(k * h) # calculate the height that need to be expanded
            imgResize = cv2.resize(imgCrop, (IMGSIZE, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((IMGSIZE - hCal)/2) # calculate the gap to center the height of imgCrop in imgWhite
            imgWhite[hGap : hCal + hGap, :] = imgResize  # put the imgCrop on the imgWhite

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)



    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"): # in order that we pressed s button
        counter += 1
        cv2.imwrite(f'{folder}/Image{time.time()}.png', imgWhite) # saves the image to the file by the name of the time
        print(counter)

