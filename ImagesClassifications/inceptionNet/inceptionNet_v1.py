import keras
from keras.api.layers import Conv2D, ReLU, MaxPooling2D, Concatenate, AvgPool2D, Dense, Dropout, \
    Flatten, GlobalAveragePooling2D
from keras import Model, Sequential, Input


def conv_block(x, _filter: int,

               kernel_size: int,
               strides: int,
               padding: str,
               name: str) -> Model:
    """
    Creates a convolutional block in a neural network model.

    Args:
        x: The input tensor to the convolutional block.
        _filter: Number of filters for the convolutional layer.
        kernel_size: Size of the convolutional kernel.
        strides: Strides for the convolution operation.
        padding: The type of padding to apply.
        name: Name for the convolutional block.

    Returns:
        Model: The output model after applying the convolutional block.
    """
    return Sequential(
        [Conv2D(_filter, kernel_size, strides, padding=padding),
         ReLU(), ],
        name=name)(x)


def inception_block(x,
                    filter_1,
                    middel_3, filter_3,
                    middel_5, filter_5,
                    filter_last, name) -> Model:
    """
    Creates an inception block in an InceptionNet model.

    Args:
        x: The input tensor to the inception block.
        filter_1: Number of filters for the 1x1 convolution.
        middel_3: Number of filters for the 1x1 convolution before the 3x3 convolution.
        filter_3: Number of filters for the 3x3 convolution.
        middel_5: Number of filters for the 1x1 convolution before the 5x5 convolution.
        filter_5: Number of filters for the 5x5 convolution.
        filter_last: Number of filters for the last 1x1 convolution.
        name: Name for the block.

    Returns:
        Model: A concatenated output of convolutional layers in the inception block.
    """
    conv_1x1 = conv_block(x, filter_1, kernel_size=1, strides=1, padding='same', name=name + "_conv_1x1")

    middle_3x3 = conv_block(x, middel_3, kernel_size=1, strides=1, padding="same", name=name + "_middel_3x3")
    conv_3x3 = conv_block(middle_3x3, filter_3, kernel_size=3, strides=1, padding="same", name=name + "_conv_3x3")

    middle_5x5 = conv_block(x, middel_5, kernel_size=1, strides=1, padding="same", name=name + "_middel_5x5")
    conv_5x5 = conv_block(middle_5x5, filter_5, kernel_size=3, strides=1, padding="same", name=name + "_conv_5x5")

    conv_pool = MaxPooling2D(pool_size=3, strides=1, padding="same", name=name + "_pool")(x)
    conv_pool = conv_block(conv_pool, filter_last, kernel_size=1, strides=1, padding="same", name=name + "_conv_last")

    return Concatenate(name=name + "_concat")([conv_1x1, conv_3x3, conv_5x5, conv_pool])


def auxiliary_block(x, num_cls, name):
    """
    Creates an InceptionNet model for image classification.

    Args:
        num_cls (int): The number of classes in the classification task.

    Returns:
        Model: The InceptionNet model with the following outputs:
            - output: The final output layer for classification.
            - auxiliary_4a: The auxiliary output layer for classification.
            - auxiliary_4d: The auxiliary output layer for classification.

    The InceptionNet model consists of several layers including convolutional layers,
    pooling layers, and inception blocks. The inception blocks are composed of
    convolutional layers with different kernel sizes and pooling layers. The model
    also includes auxiliary output layers at certain stages for better
    performance.

    The input shape of the model is (224, 224, 3), representing the size of the
    input image (224x224 pixels) with 3 color channels (RGB).

    The output layer has a softmax activation function for multi-class
    classification.

    Example usage:
        model = inception_net(num_cls=1000)
    """
    pool = AvgPool2D(pool_size=(5, 5), strides=(3, 3), name=name + "_avg_pool")(x)
    conv = conv_block(pool, 128, kernel_size=1, strides=1, padding="valid", name=name + "_conv")

    classifier = Sequential(
        [
            Flatten(),
            Dense(1024, activation="relu"),
            Dropout(0.4),
            Dense(num_cls, activation="softmax")
        ], name=name + "_classifier"
    )(conv)

    return classifier


def inception_net(num_cls):
    """Instantiates the Inception v1 architecture.

            Reference:
            - [Going Deeper with Convolutions](
            https://arxiv.org/abs/1409.4842) (CVPR 2014)

            Args:
                num_cls: optional number of classes to classify images.

            Returns:
                A model instance.
            """
    _input = Input(shape=(224, 224, 3), name="input")
    layer_1 = Sequential(
        [
            Conv2D(64, kernel_size=7, strides=2, padding="same", name="layer_1_conv1"),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            Conv2D(192, kernel_size=3, strides=1, padding="same", name="layer_1_conv2"),
            Conv2D(192, kernel_size=3, strides=1, padding="same", name="layer_1_conv3"),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")],
        name="layer_1",
    )(_input)
    inception_3a = inception_block(layer_1, 64, 96, 128, 16, 32, 32, name='inception_3a')
    inception_3b = inception_block(inception_3a, 128, 128, 192, 32, 96, 64, name='inception_3b')

    pool_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool_4")(inception_3b)

    inception_4a = inception_block(pool_4, 192, 96, 208, 16, 48, 64, name='inception_4a')
    inception_4b = inception_block(inception_4a, 160, 112, 224, 24, 64, 64, name='inception_4b')
    inception_4c = inception_block(inception_4b, 128, 128, 256, 24, 64, 64, name='inception_4c')
    inception_4d = inception_block(inception_4c, 256, 160, 320, 32, 128, 128, name='inception_4d')

    pool_5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool_5")(inception_4d)

    inception_5a = inception_block(pool_5, 256, 160, 320, 32, 128, 128, name='inception_5a')
    inception_5b = inception_block(inception_5a, 384, 192, 384, 48, 128, 128, name='inception_5b')

    auxiliary_4a = auxiliary_block(inception_4a, num_cls, name="auxiliary_4a")
    auxiliary_4d = auxiliary_block(inception_4d, num_cls, name="auxiliary_4d")

    avg_pool = GlobalAveragePooling2D()(inception_5b)
    dropout = Dropout(0.5)(avg_pool)
    output = Dense(num_cls, activation="softmax")(dropout)

    return Model(inputs=_input, outputs=[output, auxiliary_4a, auxiliary_4d])


if __name__ == "__main__":
    model = inception_net(1000)
    print(model.summary())
