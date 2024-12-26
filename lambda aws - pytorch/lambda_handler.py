import json
import cv2
import numpy as np
import requests
import boto3
import uuid
from inference import EmotionPredictor

# initialize S3 client
s3_client=boto3.client('s3')
S3_BUCKET_NAME="image-results"
inference=EmotionPredictor("./models/best_checkpoint.ckpt")

def lambda_handler(event, context):
    """ AWS lambda handler for inference
        Input: use direct url 
                { "image_url": "https://drive.google.com/uc?id=1GqISERXvrCKxtwJfMKzIKCVi93JAOeSs" 
                }
        Return: image_result_ulr
    """
    try:
        # get input from event
        print("Start reading image...")
        image_url=event.get("image_url")        
        if not image_url:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image URL provided'})
            }
        print("Input image:", image_url)

        # download the image and decode from url content
        response=requests.get(image_url)
        image_arr=np.frombuffer(response.content,np.uint8)
        image=cv2.imdecode(image_arr,cv2.IMREAD_COLOR)
        if image is None:
            return {
                'statusCode':400,
                'body': json.dumps({'error': 'Invalid image data'})
            }
        
        # process the image
        image_result=inference.inference_image(image)
        
        # encode the result image to bytes
        _, buffer=cv2.imencode('.png',image_result)
        image_bytes=buffer.tobytes()

        # generate a unique key for the processed image
        image_key = f"processed_images/{uuid.uuid4()}.png"
        # upload to s3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=image_key,
            Body=image_bytes,
            ContentType='image/png',
            #ACL='public-read' # make the image publicly accessible
        )

        # generate public url for the image
        image_result_url=f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{image_key}"
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'image_result_url': image_result_url
            })
        }
    except Exception as e:
        return {
            'statusCode':500,
            'body': json.dumps({'error': str(e)})
        }