
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    Dense
)


class CNN:

    @staticmethod
    def build(
        width,
        height,
        depth,
        total_classes,
        Saved_Weights_Path=None
    ):
        # Initialize the model
        model = Sequential()

        # Define input shape: (height, width, channels)
        model.add(Input(shape=(height, width, depth)))

        # First CONV -> RELU -> POOL layer
        model.add(
            Conv2D(
                filters=20,
                kernel_size=(5, 5),
                padding="same",
                data_format="channels_last"
            )
        )
        model.add(Activation("relu"))
        model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2)
            )
        )

        # Second CONV -> RELU -> POOL layer
        model.add(
            Conv2D(
                filters=50,
                kernel_size=(5, 5),
                padding="same",
                data_format="channels_last"
            )
        )
        model.add(Activation("relu"))
        model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2)
            )
        )

        # Third CONV -> RELU -> POOL layer
        model.add(
            Conv2D(
                filters=100,
                kernel_size=(3, 3),
                padding="same",
                data_format="channels_last"
            )
        )
        model.add(Activation("relu"))
        model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2)
            )
        )

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Output layer: one class for each digit (0-9)
        model.add(Dense(total_classes))
        model.add(Activation("softmax"))

        # Load pretrained weights if provided
        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)

        return model


# End of CNN class
