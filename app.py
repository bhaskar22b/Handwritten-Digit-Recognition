import os
import base64
import numpy as np
import cv2

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.keras")

app = Flask(
    __name__,
    static_folder="public",
    static_url_path=""
)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# Load the trained model once when the app starts.
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Make sure mnist_cnn.keras is in the project root."
    )

model = load_model(MODEL_PATH, compile=False)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(image_bytes):
    """Convert an uploaded image into the CNN's 28x28 input format."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("The uploaded image is too small.")

    # Make the background black and the digit white, like MNIST.
    border = np.concatenate(
        [image[0, :], image[-1, :], image[:, 0], image[:, -1]]
    )

    if np.mean(border) > 127:
        image = 255 - image

    # Convert the image into a binary image.
    _, threshold = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find the digit and crop away most of the empty background.
    points = cv2.findNonZero(threshold)

    if points is None:
        raise ValueError(
            "No digit was detected. Please upload a clear handwritten digit."
        )

    x, y, width, height = cv2.boundingRect(points)
    digit = threshold[y:y + height, x:x + width]

    # Resize while preserving the digit's proportions.
    scale = 20.0 / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    digit = cv2.resize(
        digit, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    # Center the digit on a black 28x28 canvas.
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = digit

    # Match the model's input shape: (1, 28, 28, 1).
    processed = canvas.astype(np.float32) / 255.0
    processed = processed.reshape(1, 28, 28, 1)

    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise ValueError("Could not prepare the image preview.")

    preview = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return processed, preview


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_preview = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("digit_image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please choose an image first."
        elif not allowed_file(uploaded_file.filename):
            error = "Please upload a PNG, JPG, JPEG, or BMP image."
        else:
            try:
                image_bytes = uploaded_file.read()

                if not image_bytes:
                    raise ValueError("The selected file is empty.")

                processed, image_preview = preprocess_image(image_bytes)
                probabilities = model.predict(processed, verbose=0)[0]

                prediction = int(np.argmax(probabilities))
                confidence = float(probabilities[prediction] * 100)

            except (ValueError, cv2.error) as exc:
                error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_preview=image_preview,
        error=error,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        image_preview=None,
        error="The image is too large. Please upload an image under 5 MB.",
    ), 413


if __name__ == "__main__":
    app.run(debug=True)