
import argparse
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD
from cnn.neural_network import CNN


# --------------------------------------------------
# 1. Parse command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Handwritten Digit Recognition using CNN"
)

parser.add_argument(
    "-s", "--save_model",
    type=int,
    default=-1,
    help="Save the trained model if set to 1"
)

parser.add_argument(
    "-l", "--load_model",
    type=int,
    default=-1,
    help="Load an existing model if set to 1"
)

parser.add_argument(
    "-w", "--save_weights",
    type=str,
    default="mnist_cnn.weights.h5",
    help="Path for saving or loading model weights"
)

args = vars(parser.parse_args())


# --------------------------------------------------
# 2. Load the MNIST dataset
# --------------------------------------------------

print("\nLoading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Convert pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNN expects (height, width, channels)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)


# --------------------------------------------------
# 3. Build the CNN model
# --------------------------------------------------

print("\nBuilding CNN model...")

clf = CNN.build(
    width=28,
    height=28,
    depth=1,
    total_classes=10
)

# Load weights if requested
if args["load_model"] > 0:
    if not os.path.exists(args["save_weights"]):
        raise FileNotFoundError(
            f"Weights file not found: {args['save_weights']}\n"
            "Train and save the model first."
        )

    clf.load_weights(args["save_weights"])
    print("Pretrained weights loaded.")


# --------------------------------------------------
# 4. Compile the model
# --------------------------------------------------

sgd = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

clf.compile(
    optimizer=sgd,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

clf.summary()


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

if args["load_model"] < 0:

    print("\nTraining the CNN model...")

    history = clf.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 6. Evaluate the model
# --------------------------------------------------

print("\nEvaluating the model...")

loss, accuracy = clf.evaluate(
    x_test,
    y_test,
    batch_size=128,
    verbose=1
)

print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 7. Save the model or weights
# --------------------------------------------------

if args["save_model"] > 0:

    clf.save_weights(args["save_weights"])

    print(
        f"Weights saved successfully: "
        f"{args['save_weights']}"
    )

# Save the complete model for convenient future use
clf.save("mnist_cnn.keras")
print("Complete model saved as mnist_cnn.keras")


# --------------------------------------------------
# 8. Predict random test images
# --------------------------------------------------

print("\nTesting random handwritten digits...")

random_indices = np.random.choice(
    len(x_test),
    size=5,
    replace=False
)

plt.figure(figsize=(12, 4))

for position, index in enumerate(random_indices):

    # Select one image with shape (1, 28, 28, 1)
    image_input = x_test[index][np.newaxis, ...]

    # Predict digit
    probabilities = clf.predict(
        image_input,
        verbose=0
    )

    predicted_digit = int(
        np.argmax(probabilities[0])
    )

    actual_digit = int(y_test[index])

    print(
        f"Predicted: {predicted_digit}, "
        f"Actual: {actual_digit}"
    )

    # Display image with OpenCV
    image = (x_test[index].squeeze() * 255).astype(
        "uint8"
    )

    image = cv2.resize(
        image,
        (140, 140),
        interpolation=cv2.INTER_LINEAR
    )

    plt.subplot(1, 5, position + 1)
    plt.imshow(image, cmap="gray")
    plt.title(
        f"Predicted: {predicted_digit}\n"
        f"Actual: {actual_digit}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# End of program
# --------------------------------------------------
