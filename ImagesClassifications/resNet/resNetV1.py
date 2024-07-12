from ImagesClassifications.resNet.baseNet import stack_residual_v1, res_net
import keras

def res_net_50(num_clss):
    """
    Instantiates the ResNet50 architecture.
    """

    def arch_builder(x):
        x = stack_residual_v1(x, 64, 3, 1, name="res_1")
        x = stack_residual_v1(x, 128, 4, 1, name="res_2")
        x = stack_residual_v1(x, 256, 6, 1, name="res_3")
        x = stack_residual_v1(x, 512, 3, 1, name="res_4")

        return x

    return res_net(
        num_clss,
        arch_builder,
        False,
        True,
        model_name="resnet_50", )


def res_net_101(num_clss):
    """Instantiates the ResNet101 architecture."""

    def arch_builder(x):
        x = stack_residual_v1(x, 64, 3, 1, name="res_1")
        x = stack_residual_v1(x, 128, 4, 1, name="res_2")
        x = stack_residual_v1(x, 256, 23, 1, name="res_3")
        x = stack_residual_v1(x, 512, 3, 1, name="res_4")

        return x

    return res_net(
        num_clss,
        arch_builder,
        False,
        True,
        model_name="resnet_101", )


def res_net_152(num_clss):
    """Instantiates the ResNet152 architecture."""

    def arch_builder(x):
        x = stack_residual_v1(x, 64, 3, 1, name="res_1")
        x = stack_residual_v1(x, 128, 8, 1, name="res_2")
        x = stack_residual_v1(x, 256, 36, 1, name="res_3")
        x = stack_residual_v1(x, 512, 3, 1, name="res_4")

        return x

    return res_net(
        num_clss,
        arch_builder,
        False,
        True,
        model_name="resnet_152", )


doc = """
Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2015)
        

Args:
    classes(int): number of classes to classify images into.
    """

setattr(res_net_50, "__doc__", res_net_50.__doc__ + doc)
setattr(res_net_101, "__doc__", res_net_101.__doc__ + doc)
setattr(res_net_152, "__doc__", res_net_152.__doc__ + doc)

if __name__ == "__main__":
    res_50 = res_net_50(1000)
    res_101 = res_net_101(1000)
    res_152 = res_net_152(1000)
    keras.utils.plot_model(res_50, show_shapes=True,
                           show_layer_names=True,
                           to_file="../images/resNet_50.png")
    keras.utils.plot_model(res_101, show_shapes=True,
                           show_layer_names=True,
                           to_file="../images/resNet_101.png")
    keras.utils.plot_model(res_152, show_shapes=True,
                           show_layer_names=True,
                           to_file="../images/resNet_152.png")
