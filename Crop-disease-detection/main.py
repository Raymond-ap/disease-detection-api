from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "exp://172.20.10.3:19000"  # expo server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# Definately not the best way to do this, but it works
PLANTMODEL = tf.keras.models.load_model("./models/version_10.keras")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img = np.array(img.resize((256, 256))) / 256
    img = np.expand_dims(img, axis=0)
    prediction = PLANTMODEL.predict(img)
    confidence = float(np.max(prediction))

    return {
        "class_name": PLANT_CLASS_NAMES[np.argmax(prediction)],
        "confidence": confidence,
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
