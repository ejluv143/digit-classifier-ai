import os
import base64
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from io import BytesIO

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_augmented_digit_classifier.keras")

model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(title="Digit Classifier AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image: str


def preprocess_canvas_image(image):
    image = image.convert("L")
    image = ImageOps.invert(image)

    img = np.array(image)

    # Remove light noise
    img[img < 50] = 0

    # Find digit bounding box
    coords = np.column_stack(np.where(img > 0))

    if coords.size == 0:
        return np.zeros((1, 784))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img = img[y_min:y_max + 1, x_min:x_max + 1]

    pil_img = Image.fromarray(img)

    # MNIST style: digit fits inside 20x20 box
    pil_img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center in 28x28 black canvas
    new_img = Image.new("L", (28, 28), 0)

    left = (28 - pil_img.width) // 2
    top = (28 - pil_img.height) // 2

    new_img.paste(pil_img, (left, top))

    img_array = np.array(new_img) / 255.0
    return img_array.reshape(1, 28, 28, 1)


@app.get("/")
def home():
    return {"message": "Digit Classifier API is running"}


@app.post("/predict")
def predict_digit(request: ImageRequest):
    image_data = request.image.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    image = Image.open(BytesIO(image_bytes)).convert("L")
    img_array = preprocess_canvas_image(image)

    prediction = model.predict(img_array, verbose=0)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    probabilities = {
        str(i): float(prob)
        for i, prob in enumerate(prediction[0])
    }

    return {
        "prediction": predicted_digit,
        "confidence": round(confidence * 100, 2),
        "probabilities": probabilities
    }