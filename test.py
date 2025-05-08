import numpy as np
import tensorflow as tf
import cv2

# Load your trained model once
model = tf.keras.models.load_model('crop_detector_model.keras')

def predict_crop_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (128, 128))  # Match your model input size
        input_arr = np.expand_dims(img_resized, axis=0)
        input_arr = input_arr / 255.0

        prediction = model.predict(input_arr)
        result = "Crop Image" if prediction[0][0] < 0.8 else "Not a Crop Image"
        return {"prediction": result, "confidence": float(prediction[0][0])}
    except Exception as e:
        return {"error": str(e)}
