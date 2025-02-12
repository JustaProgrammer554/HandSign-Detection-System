#This program has been written in python version 3.8.10 64 bits.

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO, emit # for real time client server communication
import numpy as np
import math


#Script used to stream the output on the ejs video screen.

app = Flask(__name__)
socketio = SocketIO(app)
flask_server_port = 5000

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

OFFSET = 20
IMGSIZE = 300

folder = "FolderPATH"
counter = 0

labels = ["A Sign", "B Sign", "C Sign"]

def frame_generator():

    while True:
        success, img = cap.read() # get info from the 
        if not success: #break if the frame was not successfully captured
            break
        imgOutput = img.copy()
        hands, img = detector.findHands(img) # get info from the hand positions

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((IMGSIZE, IMGSIZE, 3), np.uint8) * 255 # creating the white background for the data img
            y1 = max(0, y - OFFSET)
            y2 = min(img.shape[0], y + h + OFFSET)
            x1 = max(0, x - OFFSET)
            x2 = min(img.shape[1], x + w + OFFSET)
            imgCrop = img[y1:y2, x1:x2] # crop the hand for data img

            imgCropShape = imgCrop.shape # sizes of the imgCrop[h, w, channel]

            aspectRatio = h / w # the ratio between the height and the width of the imgCrop

            if aspectRatio > 1: # check if the img is rectangle
                k = IMGSIZE / h
                wCal = math.ceil(k * w) # calculate the width that need to be expanded
                imgResize = cv2.resize(imgCrop, (wCal, IMGSIZE))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((IMGSIZE - wCal)/2) # calculate the gap to center the width of imgCrop in imgWhite
                imgWhite[:, wGap : wCal + wGap] = imgResize  # put the imgCrop on the imgWhite
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                #print(prediction, index) 
                socketio.emit('sign_detected', {'sign': labels[index]})

            else:
                k = IMGSIZE / w
                hCal = math.ceil(k * h) # calculate the height that need to be expanded
                imgResize = cv2.resize(imgCrop, (IMGSIZE, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((IMGSIZE - hCal)/2) # calculate the gap to center the height of imgCrop in imgWhite
                imgWhite[hGap : hCal + hGap, :] = imgResize  # put the imgCrop on the imgWhite
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                socketio.emit('sign_detected', {'sign': labels[index]})

            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2) # detection text
            cv2.rectangle(imgOutput, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), (255, 0, 255), 3) # detection rectangle

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes() #
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=flask_server_port)
    socketio.run(app)