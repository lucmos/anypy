import pytest
import torch

from anypy.nn.dyncnn import infer_convolution2d, infer_transposed_convolution2d


@pytest.mark.parametrize(
    "input_shape, output_shape, kernel_size, stride, padding, dilation",
    [
        ((1, 3, 32, 32), (1, 16, 32, 32), None, (1, 1), (1, 1), (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), None, (1, 1), (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), (1, 1), None, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), (1, 1), (1, 1), None),
        #
        ((1, 3, 32, 32), (1, 1, 16, 16), None, (1, 1), (1, 1), (1, 1)),
        ((1, 3, 32, 32), (1, 1, 16, 16), (3, 3), None, (1, 1), (1, 1)),
        # ((1, 3, 32, 32), (1, 1, 16, 16), (3, 3), (1, 1), None, (1, 1)), # Impossible
        ((1, 3, 32, 32), (1, 1, 16, 16), (3, 3), (1, 1), (1, 1), None),
        #
        ((1, 3, 32, 32), (1, 1, 6, 6), None, (1, 1), (1, 1), (1, 1)),
        ((1, 3, 32, 32), (1, 1, 8, 8), (3, 3), (4, 4), (1, 1), (1, 1)),
        # ((1, 3, 32, 32), (1, 1, 6, 6), (3, 3), (1, 1), None, (1, 1)), # Impossible
        ((1, 3, 32, 32), (1, 1, 6, 6), (3, 3), (1, 1), (1, 1), None),
        #
        # ((1, 3, 32, 32), (1, 1, 100, 100), None, (1, 1), (1, 1), (1, 1)), # Impossible
        # ((1, 3, 32, 32), (1, 1,100, 100), (3, 3), None, (1, 1), (1, 1)), # Impossible
        ((1, 3, 32, 32), (1, 1, 100, 100), (3, 3), (1, 1), None, (1, 1)),
        # ((1, 3, 32, 32), (1, 1, 100, 100), (3, 3), (1, 1), (1, 1), None), # Impossible
    ],
)
def test_infer_convolution(
    input_shape,
    output_shape,
    kernel_size,
    stride,
    padding,
    dilation,
):
    x = torch.randn(input_shape)

    conv = infer_convolution2d(
        input_shape,
        output_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    y = conv(x)

    assert y.shape == output_shape


@pytest.mark.parametrize(
    "input_shape, output_shape, kernel_size, stride, padding, dilation",
    [
        ((1, 3, 32, 32), (1, 16, 32, 32), None, None, (1, 1), (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), None, None, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), (1, 1), None, None),
        ((1, 3, 32, 32), (1, 16, 32, 32), None, (1, 1), None, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), None, (1, 1), (1, 1), None),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), None, (1, 1), None),
    ],
)
def test_infer_convolution_errors(
    input_shape,
    output_shape,
    kernel_size,
    stride,
    padding,
    dilation,
):
    with pytest.raises(ValueError):
        _ = infer_convolution2d(
            input_shape,
            output_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


@pytest.mark.parametrize(
    "input_shape, output_shape, kernel_size, stride, padding, output_padding, dilation",
    [
        # kernel
        ((1, 3, 32, 32), (1, 16, 32, 32), None, 1, 0, (0, 0), (1, 1)),
        ((1, 3, 32, 32), (1, 16, 37, 37), None, 1, 0, (0, 0), (1, 1)),
        ((1, 3, 32, 32), (1, 16, 42, 42), None, 1, 0, (0, 0), (1, 1)),
        # stride
        ((1, 3, 32, 32), (1, 16, 65, 65), (3, 3), None, 0, (0, 0), (1, 1)),
        ((1, 3, 32, 32), (1, 16, 96, 96), (3, 3), None, 0, (0, 0), (1, 1)),
        # padding
        ((1, 3, 32, 32), (1, 16, 34, 34), (3, 3), (1, 1), None, (0, 0), (1, 1)),
        ((1, 3, 32, 32), (1, 16, 61, 61), (3, 3), (2, 2), None, (0, 0), (1, 1)),
        # output padding
        ((1, 3, 32, 32), (1, 16, 66, 66), (3, 3), (2, 2), 0, None, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 65, 65), (3, 3), (2, 2), 0, None, (1, 1)),
        # dilation
        ((1, 3, 32, 32), (1, 16, 65, 65), (3, 3), (2, 2), 0, 0, None),
        ((1, 3, 32, 32), (1, 16, 67, 67), (3, 3), (2, 2), 0, 0, None),
        ((1, 3, 32, 32), (1, 16, 73, 73), (3, 3), (2, 2), 0, 0, None),
    ],
)
def test_infer_transposed_convolution(
    input_shape,
    output_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
):
    x = torch.randn(input_shape)

    conv = infer_transposed_convolution2d(
        input_shape,
        output_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )

    y = conv(x)

    assert y.shape == output_shape


@pytest.mark.parametrize(
    "input_shape, output_shape, kernel_size, stride, padding, output_padding, dilation",
    [
        ((1, 3, 32, 32), (1, 16, 32, 32), None, None, (1, 1), None, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), None, None, None, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), (1, 1), None, None, None),
        ((1, 3, 32, 32), (1, 16, 32, 32), None, (1, 1), None, None, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), None, (1, 1), (1, 1), None, None),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), None, (1, 1), None, None),
        ((1, 3, 32, 32), (1, 16, 32, 32), None, None, (1, 1), 0, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), None, None, 0, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), (1, 1), None, 0, None),
        ((1, 3, 32, 32), (1, 16, 32, 32), None, (1, 1), None, 0, (1, 1)),
        ((1, 3, 32, 32), (1, 16, 32, 32), None, (1, 1), (1, 1), 0, None),
        ((1, 3, 32, 32), (1, 16, 32, 32), (3, 3), None, (1, 1), 0, None),
    ],
)
def test_infer_tranpose_convolution_errors(
    input_shape,
    output_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
):
    with pytest.raises(ValueError):
        _ = infer_transposed_convolution2d(
            input_shape,
            output_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
        )
