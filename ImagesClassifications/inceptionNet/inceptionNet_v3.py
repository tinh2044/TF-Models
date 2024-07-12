import keras
from keras import layers


def conv2d(x, filters, kernel_size, strides=1, padding='valid', name=None):
    """
    Perform a 2D convolution on the input tensor x.

    Args:
        x: Input tensor.
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers, specifying the stride of the convolution along the height and width.
        padding: One of "valid" or "same" (case-insensitive).
        name: A string, the name of the Sequential model.

    Returns:
        Sequential model with Conv2D, BatchNormalization, and ReLU Activation layers applied to the input tensor x.
    """
    return keras.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
        layers.BatchNormalization(),
        layers.Activation('relu'),
    ], name=name)(x)


def inception_a(x, _filter1x1, middle3x3, _filter3x3, middle3x3_dbl, _filter3x3_dbl, _filter_pool, name):
    """
    Instances a new Inception block, which extend from inception block of inception net and replace 5x5 convolution with
    two 3x3 convolution.

    Args:
        x (Tensor): The input tensor.
        _filter1x1 (int): The number of filters for the 1x1 convolution.
        middle3x3 (int): The number of filters for the 1x1 convolution before the 3x3 convolution.
        _filter3x3 (int): The number of filters for the 3x3 convolution.
        middle3x3_dbl (int): The number of filters for the 1x1 convolution before the double 3x3 convolution.
        _filter3x3_dbl (int): The number of filters for each of the two 3x3 convolutions.
        _filter_pool (int): The number of filters for the 1x1 convolution applied to the average-pooled tensor.
        name (str): The name of the module.

    Returns:
        Tensor: The output tensor after applying the Inception-A module.
    """
    branch_1x1 = conv2d(x, 64, (1, 1), name=name + '_1x1')

    branch_3x3 = conv2d(x, middle3x3, (1, 1), name=name + '_middle_3x3')
    branch_3x3 = conv2d(branch_3x3, _filter3x3, (3, 3), padding='same', name=name + '_3x3')

    branch_3x3_dbl = conv2d(x, middle3x3_dbl, (1, 1), name=name + '_middle_3x3_dbl')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (3, 3), padding='same', name=name + '_3x3_dbl_1')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (3, 3), padding='same', name=name + '_3x3_dbl_2')

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, _filter_pool, (1, 1), name=name + '_pool')

    return layers.Concatenate(name=name + "concat")([branch_1x1, branch_3x3, branch_3x3_dbl, branch_pool])


def reduce_inception_a(x, name):

    branch_3x3 = conv2d(x, 384, (3, 3), strides=2, name=name + '_3x3')

    branch_3x3_dbl = conv2d(x, 64, 1, 1, name=name + '_1x1')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, 96, (3, 3), padding="same", name=name + '_3x3(1)')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, 96, (3, 3), strides=2, name=name + '_3x3(2)')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), name=name + "_pool")(x)

    return layers.Concatenate(name=name + "concat")([branch_3x3, branch_3x3_dbl, branch_pool])


def inception_b(x, _filter1x1, middle7x7, _filter7x7, middle7x7_dbl, _filter7x7_dbl, _filter_pool, name):
    """
        Instances a new Inception block, which extend from inception block of inception net and replace 5x5 convolution with
        two 3x3 convolution .Finally, replace all 3x3 convolution with a kernel 1x7 and 7x1.

        Args:
            x (Tensor): The input tensor.
            _filter1x1 (int): The number of filters for the 1x1 convolution.
            middle7x7 (int): The number of filters for the 1x1 convolution and 1x7 convolution before the 7x1
            convolution.
            _filter7x7 (int): The number of filters for the 7x1 convolution.
            middle7x7_dbl (int): The number of filters for the 1x1, 1x7, 7x1 convolutions before the double last convolution.
            _filter7x7_dbl (int): The number of filters for each of the last 7x1 convolutions.
            _filter_pool (int): The number of filters for the 1x1 convolution applied to the average-pooled tensor.
            name (str): The name of the module.

        Returns:
            Tensor: The output tensor after applying the Inception-A module.
        """
    branch_1x1 = conv2d(x, 192, (1, 1), name=name + '_1x1')

    branch_7x7 = conv2d(x, middle7x7, (1, 1), name=name + '_middle_7x7')
    branch_7x7 = conv2d(branch_7x7, middle7x7, (1, 7), padding="same", name=name + '_1x7')
    branch_7x7 = conv2d(branch_7x7, _filter7x7, (7, 1), padding="same", name=name + '_7x1')

    branch_7x7_dlb = conv2d(x, middle7x7_dbl, (1, 1), name=name + '_middle_7x7_dbl')
    branch_7x7_dlb = conv2d(branch_7x7_dlb, middle7x7_dbl, (1, 7), padding="same", name=name + '_1x7(1)')
    branch_7x7_dlb = conv2d(branch_7x7_dlb, middle7x7_dbl, (7, 1), padding="same", name=name + '_7x1(1)')
    branch_7x7_dlb = conv2d(branch_7x7_dlb, middle7x7_dbl, (1, 7), padding="same", name=name + '_1x7(2)')
    branch_7x7_dlb = conv2d(branch_7x7_dlb, _filter7x7_dbl, (7, 1), padding="same", name=name + '_7x1(2)')

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, _filter_pool, (1, 1), name=name + '_pool')

    return layers.Concatenate(name=name + "concat")([branch_1x1, branch_7x7, branch_7x7_dlb, branch_pool])


def reduce_inception_b(x, name):
    branch_3x3 = conv2d(x, 192, (1, 1), name=name + "_middel_3x3")
    branch_3x3 = conv2d(branch_3x3, 320, (3, 3), 2, name=name + "_3x3")

    branch_7x7x3 = conv2d(x, 192, (1, 1), name=name + "_1x1")
    branch_7x7x3 = conv2d(branch_7x7x3, 192, (1, 7), name=name + "_1x7", padding="same")
    branch_7x7x3 = conv2d(branch_7x7x3, 192, (7, 1), name=name + "_7x1", padding="same")
    branch_7x7x3 = conv2d(branch_7x7x3, 192, (3, 3), strides=2, name=name + "_3x3(l)")

    branch_pool = layers.MaxPooling2D((3, 3), (2, 2), name=name + "_pool")(x)

    return layers.Concatenate(name=name + "_concat")([branch_3x3, branch_7x7x3, branch_pool])


def inception_c(x, _filter1x1, _filter3x3, _middle_3x3_dbl, _filter3x3_dbl, _filter_pool, name):
    
    branch_1x1 = conv2d(x, _filter1x1, (1, 1), name=name + '_1x1')

    branch_3x3 = conv2d(x, _filter3x3, (1, 1), name=name+'middle_1x1')
    branch_3x3_1x3 = conv2d(branch_3x3, _filter3x3, (1, 3), padding="same", name=name + '_1x3')
    branch_3x3_3x1 = conv2d(branch_3x3, _filter3x3, (3, 1), padding="same", name=name + '_3x1')
    branch_3x3 = layers.Concatenate(name=name + "_concat_3x3")([branch_3x3_1x3, branch_3x3_3x1])

    branch_3x3_dbl = conv2d(x, _middle_3x3_dbl, (1, 1), name=name + '_dbl_1x1')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (3, 3), padding="same", name=name + '_dbl_3x3')
    branch_3x3_1x3_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (1, 3), padding="same", name=name + "_dbl_1x3")
    branch_3x3_3x1_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (3, 1), padding="same", name=name + "_dbl_3x1")
    branch_3x3_dbl = layers.Concatenate(name=name + "_concat_3x3_dbl")([branch_3x3_1x3_dbl, branch_3x3_3x1_dbl])

    branch_pool = layers.AveragePooling2D((3, 3), (1, 1), padding="same", name=name + "_avg_pool")(x)
    branch_pool = conv2d(branch_pool, _filter_pool, (1, 1), name=name + "_conv_pool")

    return layers.Concatenate(name=name + "_concat")([branch_1x1, branch_3x3, branch_3x3_dbl, branch_pool])


def inception_net_v3(num_cls):
    """Instantiates the Inception v3 architecture.

        Reference:
        - [Rethinking the Inception Architecture for Computer Vision](
        https://arxiv.org/abs/1512.00567) (CVPR 2016)

        Args:
            num_cls: optional number of classes to classify images.

        Returns:
            A model instance.
        """
    _input = layers.Input(shape=(299, 299, 3), name="Input")

    x = conv2d(_input, 32, (3, 3), strides=2, name='conv_1')
    x = conv2d(x, 32, (3, 3), name='conv_2')
    x = conv2d(x, 64, (3, 3), strides=2, name='conv_3')
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name='avg_pool_1')(x)

    x = conv2d(x, 80, (1, 1), name='conv_4')
    x = conv2d(x, 192, (3, 3), name='conv_5')

    x = inception_a(x, 64, 48, 64, 64, 96, 32, name="inceptionA_1")
    x = inception_a(x, 64, 48, 64, 64, 96, 64, name="inceptionA_2")
    x = inception_a(x, 64, 48, 64, 64, 96, 64, name="inceptionA_3")

    x = reduce_inception_a(x, name="reduce_a")

    x = inception_b(x, 192, 128, 192, 128, 192, 192, name="inceptionB_1")
    x = inception_b(x, 192, 160, 192, 160, 192, 192, name="inceptionB_2")
    x = inception_b(x, 192, 160, 192, 160, 192, 192, name="inceptionB_3")
    x = inception_b(x, 192, 192, 192, 192, 192, 192, name="inceptionB_4")

    x = reduce_inception_b(x, name="reduce_b")

    x = inception_c(x, 320, 384, 448, 384, 192, name="inceptionC_1")
    x = inception_c(x, 320, 384, 448, 384, 192, name="inceptionC_2")

    x = layers.GlobalAveragePooling2D(name="avg_pool_2")(x)
    x = layers.Dense(num_cls, activation='softmax', name="outputs")(x)
    return keras.Model(inputs=[_input], outputs=[x], name='InceptionNetV3')


if __name__ == "__main__":
    model = inception_net_v3(1000)
    print(model.summary())
