import keras
from keras import layers


def conv_block(_filter, kernel_size, strides, padding="valid", act=None, name=None):
    """
    Creates a convolutional block consisting of Conv2D, BatchNormalization, and Activation layers.

    Args:
        _filter (int): Number of filters in the Conv2D layer.
        kernel_size (int): Size of the convolutional kernel.
        strides (int): Stride size for the convolution.
        padding (str): Type of padding to apply.
        act (str or None): Activation function to use.
        name (str or None): Name of the block.

    Returns:
        Sequential: A Keras Sequential model representing the convolutional block.
    """
    conv = keras.Sequential([

        layers.Conv2D(_filter, kernel_size=kernel_size, strides=strides, padding=padding),
        layers.BatchNormalization(),

    ], name=name)
    if act is not None:
        conv.add(layers.Activation(act))

    return conv


def res_block_v1(x, _filter, kernel_size, strides, use_shortcut=None, name=None):
    """
    Creates a ResNet version 1 block with optional shortcut connection.

    Args:
        x: Input tensor.
        _filter (int): Number of filters in the convolutional layers.
        kernel_size (int): Size of the convolutional kernel.
        strides (int): Stride size for the convolutional layers.
        use_shortcut: Flag to determine if a shortcut connection should be used.
        name: Name of the block.

    Returns:
        Tensor: Output tensor after passing through the ResNet block.
    """

    if use_shortcut:
        shortcut = conv_block(4 * _filter, kernel_size=1, strides=strides, padding="valid", name=f"{name}_shortcut")(x)

    else:
        shortcut = x

    x = conv_block(_filter, 1, strides=strides, padding="valid", act='relu', name=f"{name}_conv_1")(x)
    x = conv_block(_filter, kernel_size=kernel_size, strides=1, padding="same", act='relu', name=f"{name}conv_2")(x)
    x = conv_block(4 * _filter, kernel_size=1, strides=1, act='relu', name=f"{name}conv_3")(x)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    x = layers.ReLU(name=f"{name}_relu")(x)

    return x


def stack_residual_v1(x, _filter, num_block, strides, name):
    """
    Stacks multiple ResNet version 1 blocks to form a residual network.

    Args:
        x: Input tensor.
        _filter (int): Number of filters in the convolutional layers.
        num_block (int): Number of ResNet blocks to stack.
        strides (int): Stride size for the convolutional layers.
        name: Name of the stack.

    Returns:
        Tensor: Output tensor after passing through the stacked ResNet blocks.
    """

    x = res_block_v1(x, _filter, kernel_size=3, strides=strides, use_shortcut=True, name=f"{name}_block_1")

    for i in range(2, num_block + 1):
        x = res_block_v1(x, _filter, kernel_size=3, strides=1, use_shortcut=None, name=f"{name}_block_{i}")

    return x


def res_block_v2(x, _filter, strides, use_shortcut, name):
    """A residual block for ResNetV2.

    Args:
        x: Input tensor.
        _filter: No of filters in the bottleneck layer.
        strides: Stride of the first layer. Defaults to `1`.
        use_shortcut: Use convolution shortcut if `True`, otherwise
            use identity shortcut. Defaults to `True`
        name(optional): Name of the block

    Returns:
        Output tensor for the residual block.
    """
    x = layers.BatchNormalization(name=f"{name}_preact_bn")(x)
    x = layers.ReLU(name=f"{name}_preact_relu")(x)

    if use_shortcut:
        shortcut = conv_block(4 * _filter, kernel_size=1, strides=strides, padding="valid", name=f"{name}_shortcut")(x)

    else:
        shortcut = x

    x = conv_block(_filter, 1, strides=strides, padding="valid", act='relu', name=f"{name}_conv_1")(x)
    x = conv_block(_filter, kernel_size=3, strides=1, padding="same", act='relu', name=f"{name}conv_2")(x)
    x = conv_block(4 * _filter, kernel_size=1, strides=1, act=None, name=f"{name}conv_3")(x)

    x = layers.Add(name=f"{name}_add")([shortcut, x])

    return x


def stack_residual_v2(x, filters, blocks, strides=2, name=None):
    """A set of stacked residual v2 blocks.

    Args:
        x: Input tensor.
        filters: Number of filters in the bottleneck layer in a block.
        blocks: Number of blocks in the stacked blocks.
        strides: Stride of the first layer in the first block. Defaults to `2`.
        name: Stack label.

    Returns:
        Output tensor for the stacked blocks.
    """

    x = res_block_v2(x, filters, 1, use_shortcut=True, name=name + "_block1")
    for i in range(2, blocks):
        x = res_block_v2(x, filters, 1, use_shortcut=False, name=name + "_block" + str(i))
    x = res_block_v2(
        x, filters, strides=strides, use_shortcut=False, name=name + "_block" + str(blocks)
    )
    return x


def res_net(
        num_cls,
        arch_builder,
        preact,
        use_bias,
        model_name="resnet",
):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Args:
        num_cls: optional number of classes to classify images.
        arch_builder: A function that returns output tensor for the
            stacked residual blocks.
        preact: Whether to use pre-activation or not. `True` for ResNetV2,
            `False` for ResNet and ResNeXt.
        use_bias: Whether to use biases for convolutional layers or not.
            `True` for ResNet and ResNetV2, `False` for ResNeXt.
        model_name: Name of the model.

    Returns:
        A Model instance.
    """

    _inputs = layers.Input(shape=(224, 244, 3), name="Input")

    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=use_bias, name="Conv1")(_inputs)

    if not preact:
        x = layers.BatchNormalization(name="Conv1_bn")(x)
        x = layers.ReLU(name="Con1_relu")(x)

    x = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")(x)

    x = arch_builder(x)

    if preact:
        x = layers.BatchNormalization(name="post_bn")(x)
        x = layers.ReLU(name="post_relu")(x)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dense(num_cls, activation='softmax', name="output")(x)

    return keras.Model(inputs=[_inputs], outputs=[x], name=model_name)
