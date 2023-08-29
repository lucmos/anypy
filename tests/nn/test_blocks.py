import pytest
import torch

from anypy.nn.blocks import DeepProjection, LearningBlock, build_dynamic_encoder_decoder


@pytest.mark.parametrize("size", ((28, 28), (32, 32)))
@pytest.mark.parametrize("channels", (1, 3))
@pytest.mark.parametrize(
    "hidden_dims",
    ((16, 32, 64, 128), (32, 64, 128, 256)),
)
@pytest.mark.parametrize(
    "activation",
    ("torch.nn.GELU", "torch.nn.ReLU"),
)
@pytest.mark.parametrize(
    "batch_size, use_batch_norm",
    (
        (1, False),
        (4, True),
        (8, True),
    ),
)
@pytest.mark.parametrize("remove_encoder_last_activation", (True, False))
def test_build_dynamic_encoder_decoder(
    batch_size,
    size,
    channels,
    hidden_dims,
    activation,
    use_batch_norm,
    remove_encoder_last_activation,
):
    width, height = size
    encoder, encoder_out_shape, decoder = build_dynamic_encoder_decoder(
        width=width,
        height=height,
        n_channels=channels,
        hidden_dims=hidden_dims,
        activation=activation,
        use_batch_norm=use_batch_norm,
        remove_encoder_last_activation=remove_encoder_last_activation,
    )

    x = torch.zeros([batch_size, channels, width, height])
    assert encoder(x).shape[1:] == encoder_out_shape[1:]
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
