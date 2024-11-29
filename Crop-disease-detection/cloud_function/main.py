from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

model = None
interpreter = None
input_index = None
output_index = None

PLANT_CLASS_NAMES = [
'Pepper bell Bacterial spot',
 'Pepper bell healthy',
 'Potato Early blight',
 'Potato Late blight',
 'Potato healthy',
 'Tomato Bacterial spot',
 'Tomato Early blight',
 'Tomato Late blight',
 'Tomato Leaf Mold',
 'Tomato Septoria leaf spot',
 'Tomato Spider mites Two spotted spider mite',
 'Tomato Target Spot',
 'Tomato Tomato YellowLeaf Curl Virus',
 'Tomato Tomato mosaic virus',
 'Tomato healthy'
 ]


CLOUD_BUCKET_NAME = 'botanic-tm'



# Download the blob from the bucket.
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))


# Prediction function
def predict(request):
    global model
    global interpreter
    global input_index
    global output_index
    
    if model is None:
        download_blob(CLOUD_BUCKET_NAME,  "models/version_10.keras", "/tmp/version_10.keras",) # download the model
        
        model = tf.keras.models.load_model("/tmp/version_10.keras") # load the model

    
    image = request.files['file'].read() # read the image

    image = Image.open(BytesIO(image)).convert("RGB") # convert to RGB image
    image = np.array(image.resize((256, 256))) / 255.0 # normalize the image
    image_array = np.expand_dims(image, axis=0) # add batch dimension

    prediction = model.predict(image_array)

    return {
        "class_name": PLANT_CLASS_NAMES[np.argmax(prediction)],
        "confidence": float(np.max(prediction)),
    }


