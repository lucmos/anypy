import pytest
import torch

from anypy.nn.blocks import DeepProjection, LearningBlock, build_dynamic_encoder_decoder


@pytest.mark.parametrize("input_size", ((28, 28), (32, 32), (48, 48)))
@pytest.mark.parametrize("input_channels", (1, 3, 8, 32))
@pytest.mark.parametrize("batch_size", (1, 3, 8))
def test_build_dynamic_encoder_decoder(
    input_size,
    input_channels,
    batch_size,
):
    input_shape = (batch_size, input_channels, *input_size)
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

    decoder_layers = [
        {
            "_target_": "anypy.nn.dyncnn.infer_transposed_convolution2d",
            "input_shape": "???",
            "output_shape": (-1, 32, 14, 14),
            "kernel_size": None,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
        },
        {"_target_": "torch.nn.ReLU"},
        {
            "_target_": "torch.nn.BatchNorm2d",
            "num_features": 32,
        },
        {
            "_target_": "anypy.nn.dyncnn.infer_transposed_convolution2d",
            "input_shape": "???",
            "output_shape": (-1, 32, 28, 28),
            "kernel_size": None,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
        },
        {"_target_": "torch.nn.ReLU"},
        {
            "_target_": "torch.nn.BatchNorm2d",
            "num_features": 32,
        },
        {
            "_target_": "anypy.nn.dyncnn.infer_transposed_convolution2d",
            "input_shape": "???",
            "output_shape": "???",
            "kernel_size": None,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
        },
        {"_target_": "torch.nn.Sigmoid"},
    ]
    encoder, latents_shape, decoder = build_dynamic_encoder_decoder(
        encoder_layers_config=encoder_layers,
        decoder_layers_config=decoder_layers,
        input_shape=input_shape,
    )
    x = torch.zeros(input_shape)
    assert encoder(x).shape[1:] == latents_shape[1:] == torch.Size((16, 7, 7))
    assert decoder(encoder(x)).shape == x.shape


@pytest.mark.parametrize("batch_size", (1, 8))
@pytest.mark.parametrize("num_features", (16, 64))
def test_learning_block(batch_size, num_features):
    x = torch.zeros([batch_size, num_features])
    model = LearningBlock(num_features=num_features, dropout_p=0.5)
    assert model(x).shape == x.shape


@pytest.mark.parametrize("batch_size", (4, 8))
@pytest.mark.parametrize("in_features", (16, 64))
@pytest.mark.parametrize("out_features", (16, 64))
@pytest.mark.parametrize("num_layers", (2, 3))
@pytest.mark.parametrize("activation", (torch.nn.ReLU(), torch.nn.GELU()))
def test_deep_projection(
    batch_size,
    in_features,
    out_features,
    num_layers,
    activation,
):
    x = torch.zeros([batch_size, in_features])
    model = DeepProjection(
        in_features=in_features,
        out_features=out_features,
        dropout=0.5,
        num_layers=num_layers,
        activation=activation,
    )
    out = model(x)
    assert out.shape == (batch_size, out_features)
