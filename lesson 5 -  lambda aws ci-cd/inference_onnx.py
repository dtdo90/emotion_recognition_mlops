import cv2, torch
import face_recognition
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse  # handle command-line arguments
import time

from utils import timing
import onnxruntime as ort
import logging

# Configure logging before ONNX Runtime initialization
logging.basicConfig(level=logging.INFO)

# Explicitly set up ONNX Runtime logging
try:
    ort.set_default_logger_severity(ort.LoggingLevel.WARNING)
except Exception as e:
    logging.error(f"ONNX Runtime logging setup error: {e}")

class EmotionPredictor:
    def __init__(self,model_onnx_path):
       
        self.emotions = {
            0: ['Angry', (0,0,255), (255,255,255)],
            1: ['Disgust', (0,102,0), (255,255,255)],
            2: ['Fear', (255,255,153), (0,51,51)],
            3: ['Happy', (153,0,153), (255,255,255)],
            4: ['Sad', (255,0,0), (255,255,255)],
            5: ['Surprise', (0,255,0), (255,255,255)],
            6: ['Neutral', (160,160,160), (255,255,255)]
        }
        # Set up ONNX runtime session
        print("Loading ONNX model!")
        self.ort_session = ort.InferenceSession(
            model_onnx_path, 
            providers=['CPUExecutionProvider']
        )
        print("ONNX model loaded successfully!")
        
        # set up transform
        cut_size=44
        self.transform=transforms.Compose([transforms.TenCrop(cut_size),
                                           transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                           ])
    @timing
    def inference_image(self, image):
        # Call the undecorated version internally
        return self._inference_image_internal(image)

    def _inference_image_internal(self, image):
        # This method has the core logic of inference_image without @timing
        locations = face_recognition.face_locations(image)
        for (top, right, bottom, left) in locations:
            # Prediction on face
            face = image[top:bottom, left:right, :]
            pred = self.prediction(face)

            # Draw rectangle and put text
            cv2.rectangle(image, (left, top), (right, bottom), self.emotions[pred][1], 2)
            cv2.rectangle(image, (left, top - 30), (right, top), self.emotions[pred][1], -1)
            cv2.putText(image, f'{self.emotions[pred][0]}', (left, top - 5), 0, 0.7, self.emotions[pred][2], 2)
        return image
        
    def inference_video(self,video_path):
        cap=cv2.VideoCapture(video_path)
        width,height=int(cap.get(3)), int(cap.get(4))
        fps=cap.get(cv2.CAP_PROP_FPS)
        num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {num_frames} frames | Resolution: {width}x{height}")

        # codec to write new video
        out=cv2.VideoWriter(video_path[:-4]+"_out.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
        
        with tqdm(total=num_frames, desc="Processing video frames") as pbar:
            # read frame, make inference and write into a video
            while cap.isOpened():
                success,frame=cap.read()

                if not success:
                    print("Failed to read frame!")
                    break
                # process frame
                frame=self._inference_image_internal(frame)
                out.write(frame)
                pbar.update(1)
               
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def inference_web(self,camera_idx):
        # load webcam
        cap=cv2.VideoCapture(camera_idx)

        while True:
            ret, frame=cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            # resize 1/4 of original frame for faster inference
            small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            face_locations=face_recognition.face_locations(small_frame)

            # give predictions to facial regions and display results
            for (top,right,bottom,left) in face_locations:

                face=small_frame[top:bottom,left:right,:]
                pred=self.prediction(face)

                # scale back up face locations to draw boxes
                top, right, bottom, left=top*4, right*4, bottom*4, left*4

                # draw rectangle and put texts around the faces
                cv2.rectangle(frame,(left,top), (right,bottom),self.emotions[pred][1],2)
                cv2.rectangle(frame,(left,top-50), (right,top),self.emotions[pred][1],-1)
                cv2.putText(frame,
                            f'{self.emotions[pred][0]}', 
                            (left,top-5), 0, 1.5,self.emotions[pred][2],2)

            # show frame
            cv2.imshow("Frame",frame)

            # set up an escape key
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
    def prediction(self,face): # face = face image
        # convert to gray-scale and resize it to (48,48)
        gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,(48,48)).astype(np.uint8)    # [1,48,48]

        # randomly crop the face image into 10 subimages
        inputs=self.transform(Image.fromarray(gray))      # [10,1,44,44]
        # Convert to NumPy array 
        inputs_np = inputs.numpy().astype(np.float32)

        # run inference with onnx runtime
        ort_inputs={self.ort_session.get_inputs()[0].name: inputs_np}
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        # prediction
        pred=np.mean(ort_outs[0], axis=0)                            # [7,]
        return pred.argmax()

if __name__=="__main__":
    """ To run: python inference_onnx.py --image_path path_to_image
    """
    # load predictor class
    predictor=EmotionPredictor("./models/trained_model.onnx")

    # set up argument parser
    parser=argparse.ArgumentParser(description="Emotion Predictor")
    parser.add_argument('--image_path', type=str, help="Path to the image for inference")
    parser.add_argument('--video_path', type=str, help="Path to the video for inference")
    parser.add_argument('--camera_idx', type=int, default=0, help="Webcam index for webcam inference")
    
    args = parser.parse_args()

    if args.image_path:
        image = cv2.imread(args.image_path)
        result = predictor.inference_image(image)
        cv2.imshow("Result", result)
        # close window when pressing esc
        while True:
            if cv2.waitKey(1) & 0xFF==27:
                break

    elif args.video_path:
        start=time.time()
        predictor.inference_video(args.video_path)
        end=time.time()
        print(f"Process take {end-start} seconds")
        
    elif args.camera_idx is not None:
         predictor.inference_web(args.camera_idx)  
          
    else:
        print("Please enter a valid option!")
