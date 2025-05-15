from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('trained_model.keras')

# Create FastAPI app
app = FastAPI()

class_name = ['crop',
              'not_crop']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image from the uploaded file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    image = image.resize((128, 128))
    
    # Convert the image to a numpy array
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Make it a batch of one image

    # Model prediction
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    model_prediction = class_name[result_index]
    
    return {"prediction": model_prediction}
