import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import torch
from omegaconf import OmegaConf
from torch import nn

pylogger = logging.getLogger(__name__)


def _ensure_ntuple(val: Optional[Tuple[int, int]], n: int) -> Optional[Tuple[int, int]]:
    """Convert and ensure tuple of size n.

    Args:
        val (tuple): The value to convert.
        n (int): The size of the tuple.

    Returns:
        tuple: The converted value.
    """
    if val is None:
        return None
    if isinstance(val, int):
        return (val,) * n
    if len(val) != n:
        raise ValueError(f"The length of the tuple must be equal to {n}.")
    return val


def compute_conv2d_output_shape(
    input_shape: Tuple[int, int, int, int],
    kernel_size: Optional[Tuple[int, int]] = None,
    stride: Optional[Tuple[int, int]] = None,
    padding: Optional[Tuple[int, int]] = None,
    dilation: Optional[int] = None,
) -> Tuple[int, int]:
    """Computes the output shape of a 2D convolution.

    Args:
        input_shape (Tuple[int, int, int, int]): The shape of the input tensor.
        kernel_size (Optional[Tuple[int, int]], optional): The size of the kernel. Defaults to None.
        stride (Optional[Tuple[int, int]], optional): The stride of the convolution. Defaults to None.
        padding (Optional[Tuple[int, int]], optional): The padding of the convolution. Defaults to None.
        dilation (Optional[int], optional): The dilation of the convolution. Defaults to None.

    Returns:
        Tuple[int, int]: The shape of the output tensor that a convolution would produce with these parameters.
    """
    kernel_size = _ensure_ntuple(kernel_size, n=2)
    stride = _ensure_ntuple(stride, n=2)
    padding = _ensure_ntuple(padding, n=2)
    dilation = _ensure_ntuple(dilation, n=2)

    _, _, in_height, in_width = input_shape

    resulting_shape_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    resulting_shape_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1

    return int(resulting_shape_height), int(resulting_shape_width)


def compute_tranpose_conv2d_output_shape(
    input_shape: Tuple[int, int, int, int],
    kernel_size: Optional[Tuple[int, int]] = None,
    stride: Optional[Tuple[int, int]] = None,
    padding: Optional[Tuple[int, int]] = None,
    output_padding: Optional[Tuple[int, int]] = None,
    dilation: Optional[int] = None,
) -> Tuple[int, int]:
    """Computes the output shape of a transposed 2D convolution.

    Args:
        input_shape (Tuple[int, int, int, int]): The shape of the input tensor.
        kernel_size (Optional[Tuple[int, int]], optional): The size of the kernel. Defaults to None.
        stride (Optional[Tuple[int, int]], optional): The stride of the convolution. Defaults to None.
        padding (Optional[Tuple[int, int]], optional): The padding of the convolution. Defaults to None.
        output_padding (Optional[Tuple[int, int]], optional): The output padding of the convolution. Defaults to None.
        dilation (Optional[int], optional): The dilation of the convolution. Defaults to None.

    Returns:
        Tuple[int, int]: The shape of the output tensor that a convolution would produce with these parameters.
    """
    kernel_size = _ensure_ntuple(kernel_size, n=2)
    stride = _ensure_ntuple(stride, n=2)
    padding = _ensure_ntuple(padding, n=2)
    output_padding = _ensure_ntuple(output_padding, n=2)
    dilation = _ensure_ntuple(dilation, n=2)

    _, _, in_height, in_width = input_shape

    resulting_shape_height = (
        (in_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    )
    resulting_shape_width = (
        (in_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
    )
    return (int(resulting_shape_height), int(resulting_shape_width))


def infer_convolution2d(
    input_shape: Tuple[int, int, int, int],
    output_shape: Tuple[int, int, int, int],
    kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
    stride: Optional[Union[int, Tuple[int, int]]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = 0,
    dilation: Optional[Union[int, Tuple[int, int]]] = 1,
    **kwargs,
) -> nn.Conv2d:
    """Infer the missing parameter of a convolution operation.

    The shape convention is HxW, not WxH.

    Args:
        input_shape (tuple): The shape of the input tensor.
        output_shape (tuple): The shape of the output tensor.
        kernel_shape (tuple): The shape of the kernel tensor.
        stride (tuple): The stride of the convolution.
        padding (tuple): The padding of the convolution.
        dilation (tuple): The dilation of the convolution.

    Returns:
        The convolution instantiated with the inferred parameters.
    """
    if len(input_shape) != 4:
        raise ValueError(f"The input shape must be a 4-tuple but {input_shape} was given.")
    if len(output_shape) != 4:
        raise ValueError(f"The output shape must be a 4-tuple but {output_shape} was given.")

    kernel_size = _ensure_ntuple(kernel_size, n=2)
    stride = _ensure_ntuple(stride, n=2)
    padding = _ensure_ntuple(padding, n=2)
    dilation = _ensure_ntuple(dilation, n=2)

    _, in_channels, in_height, in_width = input_shape
    _, out_channels, out_height, out_width = output_shape

    # Ensure that only one parameter is missing.
    if sum(x is None for x in [kernel_size, stride, padding, dilation]) > 1:
        raise ValueError("Only one parameter can be missing and automatically inferred.")

    if kernel_size is None:
        kernel_height = (((out_height - 1) * stride[0] - in_height - 2 * padding[0] + 1) / (-dilation[0])) + 1
        kernel_width = (((out_width - 1) * stride[1] - in_width - 2 * padding[1] + 1) / (-dilation[1])) + 1
        kernel_size = (int(kernel_height), int(kernel_width))
    elif stride is None:
        stride_height = (in_height + 2 * padding[0] - dilation[0] + (kernel_size[0] - 1)) / (out_height - 1)
        stride_width = (in_width + 2 * padding[1] - dilation[1] + (kernel_size[1] - 1)) / (out_width - 1)
        stride = (int(stride_height), int(stride_width))
    elif padding is None:
        padding_height = ((out_height - 1) * stride[0] - in_height + dilation[0] * (kernel_size[0] - 1) + 1) / 2
        padding_width = ((out_width - 1) * stride[1] - in_width + dilation[1] * (kernel_size[1] - 1) + 1) / 2
        padding = (int(padding_height), int(padding_width))
    elif dilation is None:
        dilation_height = -((out_height - 1) * stride[0] - in_height - 2 * padding[0] + 1) / (kernel_size[0] - 1)
        dilation_width = -((out_width - 1) * stride[1] - in_width - 2 * padding[1] + 1) / (kernel_size[1] - 1)
        dilation = (int(dilation_height), int(dilation_width))

    # Check if the analytic shape is the same as the expected shape.
    resulting_shape = compute_conv2d_output_shape(
        input_shape=input_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if resulting_shape != (out_height, out_width):
        raise ValueError(
            f"The resulting spatial shape of the convolution is {resulting_shape} but {(out_height, out_width)} was expected."
        )

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        **kwargs,
    )


def infer_transposed_convolution2d(
    input_shape: Tuple[int, int, int, int],
    output_shape: Tuple[int, int, int, int],
    kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
    stride: Optional[Union[int, Tuple[int, int]]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = 0,
    output_padding: Optional[Union[int, Tuple[int, int]]] = 0,
    dilation: Optional[Union[int, Tuple[int, int]]] = 1,
    **kwargs,
) -> nn.ConvTranspose2d:
    """Infer the missing parameter of a convolution operation.

    The shape convention is HxW, not WxH.

    Args:
        input_shape (tuple): The shape of the input tensor.
        output_shape (tuple): The shape of the output tensor.
        kernel_shape (tuple): The shape of the kernel tensor.
        stride (tuple): The stride of the convolution.
        padding (tuple): The padding of the convolution.
        output_padding (tuple): The output padding of the convolution.
        dilation (tuple): The dilation of the convolution.

    Returns:
        The convolution instantiated with the inferred parameters.
    """
    if len(input_shape) != 4:
        raise ValueError(f"The input shape must be a 4-tuple but {input_shape} was given.")
    if len(output_shape) != 4:
        raise ValueError(f"The output shape must be a 4-tuple but {output_shape} was given.")

    kernel_size = _ensure_ntuple(kernel_size, n=2)
    stride = _ensure_ntuple(stride, n=2)
    padding = _ensure_ntuple(padding, n=2)
    output_padding = _ensure_ntuple(output_padding, n=2)
    dilation = _ensure_ntuple(dilation, n=2)

    _, in_channels, in_height, in_width = input_shape
    _, out_channels, out_height, out_width = output_shape

    # Ensure that only one parameter is missing.
    if sum(x is None for x in [kernel_size, stride, padding, output_padding, dilation]) > 1:
        raise ValueError(
            f"Only one parameter can be missing and automatically inferred ({kernel_size=}, {stride=}, {padding=}, {output_padding=}, {dilation=})."
        )

    if kernel_size is None:
        kernel_height = (
            (out_height - (in_height - 1) * stride[0] + 2 * padding[0] - output_padding[0] - 1) / dilation[0]
        ) + 1
        kernel_width = (
            (out_width - (in_width - 1) * stride[1] + 2 * padding[1] - output_padding[1] - 1) / dilation[1]
        ) + 1
        kernel_size = (int(kernel_height), int(kernel_width))
    elif stride is None:
        stride_height = (out_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - output_padding[0] - 1) / (
            in_height - 1
        )
        stride_width = (out_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - output_padding[1] - 1) / (
            in_width - 1
        )
        stride = (round(stride_height), round(stride_width))  # fixme: is the round correct here?
    elif padding is None:
        padding_height = (
            -(out_height - (in_height - 1) * stride[0] - dilation[0] * (kernel_size[0] - 1) - output_padding[0] - 1) / 2
        )
        padding_width = (
            -(out_width - (in_width - 1) * stride[1] - dilation[1] * (kernel_size[1] - 1) - output_padding[1] - 1) / 2
        )
        padding = (int(padding_height), int(padding_width))
    elif output_padding is None:
        output_padding_height = (
            out_height - (in_height - 1) * stride[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
        )
        output_padding_width = (
            out_width - (in_width - 1) * stride[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
        )
        output_padding = (int(output_padding_height), int(output_padding_width))
    elif dilation is None:
        dilation_height = (out_height - (in_height - 1) * stride[0] + 2 * padding[0] - output_padding[0] - 1) / (
            kernel_size[0] - 1
        )
        dilation_width = (out_width - (in_width - 1) * stride[1] + 2 * padding[1] - output_padding[1] - 1) / (
            kernel_size[1] - 1
        )
        dilation = (int(dilation_height), int(dilation_width))

    # Check if the analytic shape is the same as the expected shape.
    resulting_shape = compute_tranpose_conv2d_output_shape(
        input_shape=input_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )
    if resulting_shape != (out_height, out_width):
        raise ValueError(
            f"The resulting spatial shape of the convolution is {resulting_shape} but {(out_height, out_width)} was expected."
        )

    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        **kwargs,
    )


def build_nn(
    nn_config: List[Dict[str, Any]],
    input_shape: Tuple[int, ...],
    output_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[nn.Module, torch.Size]:
    """Builds a neural network from a list of layer configurations.

    The layer configuration are instantiated with hydra.utils.instantiate.
    The input_shape and output_shape is injected into the layer configuration when it is missing "???",
    following the OmegaConf conventions.

    Example of a valid `nn_config`:

    ```python
    encoder_layers = [
        {
            "_target_": "anypy.nn.dyncnn.infer_convolution2d",
            "input_shape": "???",
            "output_shape": (-1, 32, 28, 28),
            "kernel_size": None,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
        },
        {"_target_": "torch.nn.ReLU"},
        {
            "_target_": "torch.nn.BatchNorm2d",
            "num_features": 32,
        },
        {
            "_target_": "anypy.nn.dyncnn.infer_convolution2d",
            "input_shape": "???",
            "output_shape": (-1, 32, 14, 14),
            "kernel_size": 4,
            "stride": 2,
            "padding": None,
            "dilation": 1,
        },
        {"_target_": "torch.nn.ReLU"},
        {
            "_target_": "anypy.nn.dyncnn.infer_convolution2d",
            "input_shape": "???",
            "output_shape": (-1, 16, 7, 7),
            "kernel_size": None,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
        },
        {"_target_": "torch.nn.ReLU"},
        {
            "_target_": "torch.nn.BatchNorm2d",
            "num_features": 16,
        },
    ]
    ```

    Args:
        nn_config (List[Dict[str, Any]]): The list of layer configurations in hydra format.
        input_shape (Tuple[int, ...]): The input shape of the neural network.
        output_shape (Optional[Tuple[int, ...]], optional): The output shape of the neural network. Defaults to None.

    Returns:
        Tuple[nn.Module, torch.Size]: The neural network and the output shape.
    """
    # OmegaConf does not support torch.Size sequences
    input_shape = list(input_shape)
    if output_shape is not None:
        output_shape = list(output_shape)

    layers = nn.ModuleList()

    for layer_config in nn_config:
        layer_config = OmegaConf.create(layer_config)

        if OmegaConf.is_missing(layer_config, "input_shape"):
            layer_config["input_shape"] = input_shape

        if output_shape is not None and OmegaConf.is_missing(layer_config, "output_shape"):
            layer_config["output_shape"] = output_shape

        layers.append(hydra.utils.instantiate(layer_config))

        if "output_shape" in layer_config and not OmegaConf.is_missing(layer_config, "output_shape"):
            input_shape = layer_config["output_shape"]

    return nn.Sequential(*layers), torch.Size(input_shape)
