from datetime import datetime
import cv2
import os
from keras.constraints import get
from mtcnn.mtcnn import MTCNN
from math import ceil
from imutils.video import FileVideoStream
from imutils.video import FPS
from numpy.core.defchararray import isdecimal
from utils_fr import umeyama
from resources import SupportMethods
import tensorflow as tf
import numpy as np

from fastapi import FastAPI, Response
from fastapi import responses
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse

from starlette.requests import Request



app = FastAPI(tilte = "Video Camera stream over HTML",
                description="Streaming video camera feed over a web-browser")
templates = Jinja2Templates(directory="templates/")

mtcnn = MTCNN()
feature_net = SupportMethods.build_rface()

# Test prediction for activating feature_net
test_img_ = np.full( (112,112, 3), 100, dtype="uint8" )
test_img_ = test_img_[ np.newaxis, ...]
feature_net.predict( [test_img_] )

# cap = FileVideoStream(0).start() --> this is multithreaded streamer where reading and frame processing is done in different thread
cap = cv2.VideoCapture(0)

def gen_frames():

    skip_frames = 2
    frame_count = 0

    # while cap.more():
    while True:
        ret, frame = cap.read()
        
        # Incrementing frame count
        frame_count+=1
        if frame_count == 100:
            frame_count=0

        f_strt = datetime.now()
        
        if frame_count%skip_frames==0: # skip every 3rd frame for face detection
            pass
        else:
            frame = detect_face(image=frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        print("FPS:", 1/(datetime.now() - f_strt).total_seconds())

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect_face(image):
    """
    detects faces in given image
    """
    orig_image = image.copy()
    orig_shape = image.shape
    scale_factor = 11
    new_shape = (ceil(orig_shape[0]/scale_factor), ceil(orig_shape[1]/scale_factor))

    print("original Shape:", orig_shape, "\tScaled Shape:", new_shape)
    image = cv2.resize(image, new_shape[::-1] )
    
    print(image.shape)
    faces = mtcnn.detect_faces(image)
    for face in faces:
        # print(face)
        scaled_x, scaled_y, scaled_width, scaled_height = face['box']
        
        x, y, width, height= int(scaled_x*scale_factor), int(scaled_y*scale_factor), int(scaled_width*scale_factor), int(scaled_height*scale_factor) 
        
        cropped_face = orig_image[y:y+height, x:x+width]
        # cv2.imwrite("test_output/test.jpg", cropped_face)
        f1 = get_features(cropped_face)
        # print(f1)
        
        cv2.rectangle(orig_image, (x,y), (x+width,y+height), color = (0,0,255))

    return orig_image

def get_features(cropped_face):
    '''
    cropped_face: numpy array of cropped face from image 
    return: 1D feature vector of (512,) dimension
    '''

    input_array = cropped_face[np.newaxis, ...]
    # with tf_graph.as_default():
    feature_vector = np.asarray( feature_net.predict( [input_array]), dtype='float32')
    
    return feature_vector



@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    start_cam()
    return templates.TemplateResponse("index.html", {"request":request})

@app.get("/video-feed")
def video_feed():
    return StreamingResponse( content=gen_frames(), media_type= "multipart/x-mixed-replace; boundary=frame" )

@app.get("/stop-feed")
def stop_cam():
    global cap
    cap.release()
    cap = None
    return {"CameraStopped":"Camera Stopped Successfully"}

@app.get("/start-feed")
def start_cam():
    global cap
    print(cap)
    if cap is None:
        cap = cv2.VideoCapture(0)
        return {"CameraStarted":"Camera Started Successfully"}
    else:
        return {"CameraStarted":"Camera already running"}       

