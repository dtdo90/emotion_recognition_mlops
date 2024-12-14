from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO

from inference_onnx import EmotionPredictor
import os, cv2

# initialize predictor with onnx model
predictor=EmotionPredictor("./models/trained_model.onnx")

app=FastAPI(title="MLOps Emotion Recognition")


# home page
@app.get("/")
async def home_page():
    return "<h2> Sample prediction API </h2>"

@app.get("/predict")
async def get_prediction(image_path: str=None, video_path: str=None, camera_idx: int=None):
    if image_path:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        # read image
        image = cv2.imread(image_path)
        image_result = predictor.inference_image(image)
        
        # Convert the processed image to PNG format
        _, buffer = cv2.imencode('.png', image_result)
        image_stream = BytesIO(buffer)

        # Return the image as a streaming response
        return StreamingResponse(image_stream, media_type="image/png")
         
    elif video_path:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Video not found")
        predictor.inference_video(video_path)
        video_result_path=video_path[:-4]+"_out.mp4"
        return {"video_result_path": video_result_path}
    
    elif camera_idx is not None:
        if not isinstance(camera_idx,int) or camera_idx<0:
            raise HTTPException(status_code=400, detail="Invalid index. Enter 0 or 1.")
        predictor.inference_web(camera_idx)
        return {"message": "Webcam inference started. Press esc to escape."}

       