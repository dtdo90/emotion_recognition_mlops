from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from io import BytesIO

from inference_onnx import EmotionPredictor
import cv2
import requests
import numpy as np

# initialize predictor with onnx model
predictor=EmotionPredictor("./models/trained_model.onnx")

app=FastAPI(title="MLOps Emotion Recognition")


# home page
@app.get("/")
async def home_page():
    return "<h2> Sample prediction API </h2>"

@app.get("/predict")
async def get_prediction(image_url: str=None, video_url: str=None, camera_url: int=None):
    """ image_path = https://drive.google.com/uc?id=1GqISERXvrCKxtwJfMKzIKCVi93JAOeSs """
    if image_url:
        # download the image
        response=requests.get(image_url)
        # decode image from url content
        image_arr=np.frombuffer(response.content,np.uint8)
        image=cv2.imdecode(image_arr,cv2.IMREAD_COLOR)
        image_result = predictor.inference_image(image)

        # Convert the processed image to PNG format
        _, buffer = cv2.imencode('.png', image_result)
        image_stream = BytesIO(buffer)

        # Return the image as a streaming response
        return StreamingResponse(image_stream, media_type="image/png")
         
    else:
        print("Option is not supported yet!")
       