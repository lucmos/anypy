import pytest
import torch

from anypy.nn.dynccn import infer_convolution


@pytest.mark.parametrize(
    "input_shape, expected_output_shape, kernel_size, stride, padding, dilation",
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
    expected_output_shape,
    kernel_size,
    stride,
    padding,
    dilation,
):
    x = torch.randn(input_shape)

    conv = infer_convolution(
        input_shape,
        expected_output_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    print(conv)

    y = conv(x)

    assert y.shape == expected_output_shape


@pytest.mark.parametrize(
    "input_shape, expected_output_shape, kernel_size, stride, padding, dilation",
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
    expected_output_shape,
    kernel_size,
    stride,
    padding,
    dilation,
):
    with pytest.raises(ValueError):
        _ = infer_convolution(
            input_shape,
            expected_output_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
