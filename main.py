#This program has been written in python version 3.8.10 64 bits.

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from flask_socketio import SocketIO, emit # for real time client server communication
from flask import Flask, Response, render_template, jsonify
import numpy as np
import math
import serial
import time

app = Flask(__name__)
flask_server_port = 5000
socketio = SocketIO(app)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

OFFSET = 20
IMGSIZE = 300

start_time = None
detected_sign_index = None

folder = "FolderPATH"
counter = 0

labels = ["A Sign", "B Sign", "C Sign"]
arduino_data = {'a':97, 'b':98, 'c':99} # ascii codes for 'a', 'b', 'c'
shown_data = list(arduino_data.keys())

start_time = None
detected_sign_index = None

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

            else:
                k = IMGSIZE / w
                hCal = math.ceil(k * h) # calculate the height that need to be expanded
                imgResize = cv2.resize(imgCrop, (IMGSIZE, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((IMGSIZE - hCal)/2) # calculate the gap to center the height of imgCrop in imgWhite
                imgWhite[hGap : hCal + hGap, :] = imgResize  # put the imgCrop on the imgWhite

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            if prediction[index] > 0.9999:
                socketio.emit('sign_detected', {'sign': labels[index]})
                socketio.emit('data_sent', {'commData': shown_data[index]})
                cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2) # detection text

            cv2.rectangle(imgOutput, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), (255, 0, 255), 3) # detection rectangle
            signal_control(index)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes() 
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def send_Signal(message):
    global arduino_data
    ascii_codes = list(arduino_data.values())
    ser = serial.Serial("/dev/rfcomm0", 9600, timeout=1)
    time.sleep(2)
    print("Connected to /dev/rfcomm0")
    message = int(message)
    if message >= 0 and message < len(ascii_codes):
        signal = chr(ascii_codes[message])
        ser.write(signal.encode())
    else:
        pass

    print("Message sent: ", ascii_codes[message])

def signal_control(current_index):
    global start_time, detected_sign_index
    current_time = time.time()

    if detected_sign_index != current_index:
        detected_sign_index = current_index
        start_time = current_time
        return

    if start_time and (current_time - start_time) >= 5:
        send_Signal(current_index)
        start_time = current_time + 9999

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, host='127.0.0.1', port=flask_server_port)
