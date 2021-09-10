from datetime import datetime
from fastapi import FastAPI, Response
from fastapi import responses
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import os
from mtcnn.mtcnn import MTCNN
from math import ceil
from imutils.video import FileVideoStream
from imutils.video import FPS

from starlette.requests import Request



app = FastAPI(tilte = "Video Camera stream over HTML",
                description="Streaming video camera feed over a web-browser")
templates = Jinja2Templates(directory="templates/")
mtcnn = MTCNN()

cap = cv2.VideoCapture(0)


def gen_frames():

    while True:
        success, frame = cap.read()
        f_strt = datetime.now()
        if not success:
            break
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
    scale_factor = 10
    new_shape = (ceil(orig_shape[0]/scale_factor), ceil(orig_shape[1]/scale_factor))

    print("original Shape:", orig_shape, "\tScaled Shape:", new_shape)
    image = cv2.resize(image, new_shape[::-1] )
    
    print(image.shape)
    faces = mtcnn.detect_faces(image)
    for face in faces:
        # print(face)
        scaled_x, scaled_y, scaled_width, scaled_height = face['box']
        x, y, width, height= int(scaled_x*scale_factor), int(scaled_y*scale_factor), int(scaled_width*scale_factor), int(scaled_height*scale_factor) 
        
        cv2.rectangle(orig_image, (x,y), (x+width,y+height), color = (0,0,255))

    return orig_image


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.get("/video-feed")
def video_feed():
    return StreamingResponse( content=gen_frames(), media_type= "multipart/x-mixed-replace; boundary=frame" )



