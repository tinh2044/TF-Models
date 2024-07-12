import keras
from keras import layers


def vgg_16(classes):
    """Instantiates the VGG16 model.

    Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)


    Args:
        classes: number of classes to classify images.

    Returns:
        A model instance.
    """
    _input = layers.Input(shape=(224, 224, 3))
    # Block 1
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(_input)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv1"
    )(x)
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv1"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv2"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    # Classification block
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(4096, activation="relu", name="fc1")(x)
    x = layers.Dense(4096, activation="relu", name="fc2")(x)

    x = layers.Dense(classes, activation="softmax", name="outputs")(x)

    return keras.Model(inputs=[_input], outputs=[x], name="VGG16")


if __name__ == "__main__":
    model = vgg_16(1000)
    keras.utils.plot_model(model, show_shapes=True,
                           show_layer_names=True,
                           to_file="../images/VGG_16.png")
