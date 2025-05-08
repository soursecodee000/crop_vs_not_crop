from flask import Flask, request, jsonify
from test import predict_crop_image
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Crop Image Detection API is running."

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    image = request.files['image']
    image_path = os.path.join("temp", image.filename)
    os.makedirs("temp", exist_ok=True)
    image.save(image_path)

    result = predict_crop_image(image_path)

    # Clean up the saved image
    os.remove(image_path)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
