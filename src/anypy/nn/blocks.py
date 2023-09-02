import logging
from typing import Dict, Sequence, Tuple

import torch
from torch import nn

from anypy.nn.dyncnn import build_nn

pylogger = logging.getLogger(__name__)


class LearningBlock(nn.Module):
    def __init__(self, num_features: int, dropout_p: float):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)

        self.ff = nn.Sequential(
            nn.Linear(num_features, 4 * num_features),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(4 * num_features, num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = self.norm1(x)
        x_transformed = self.ff(x_normalized)
        return self.norm2(x_transformed + x_normalized)


class DeepProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        num_layers: int = 5,
        activation: torch.nn.modules.activation = None,
    ):
        super().__init__()
        projection_inputs = [int(in_features // (2**in_dim)) for in_dim in range(0, num_layers)]
        projection_outputs = projection_inputs[1:] + [out_features]

        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = activation
        self.projection_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_dim, output_dim)
                for input_dim, output_dim in zip(projection_inputs, projection_outputs)
            ]
        )
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(input_dim) for input_dim in projection_inputs])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        for projection_layer, batch_norm in zip(self.projection_layers, self.batch_norms):
            data = batch_norm(data)
            if self.activation is not None:
                data = self.activation(data)
            data = self.dropout(data)
            data = projection_layer(data)
        return data


def build_dynamic_encoder_decoder(
    input_shape: Sequence[int],
    encoder_layers_config: Sequence[Dict],
    decoder_layers_config: Sequence[Dict],
) -> Tuple[nn.Module, Sequence[int], nn.Module]:
    """Builds a dynamic convolutional encoder-decoder model that accepts the given input shape.

    The missing parameters are inferred from the input shape and the given configuration.
    See also: `build_nn` for a more general version of this function.

    Example of a AE configuration:
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
    ```

    Args:
        input_shape: the input shape of the model.
        encoder_layers_config: the configuration of the encoder layers.
        decoder_layers_config: the configuration of the decoder layers.

    Returns:
        the encoder model, the shape in the latent space,  the decoder module
    """
    encoder, encoder_output_shape = build_nn(
        encoder_layers_config,
        input_shape=input_shape,
    )

    decoder, decoder_output_shape = build_nn(
        decoder_layers_config,
        input_shape=encoder_output_shape,
        output_shape=input_shape,
    )
    assert tuple(input_shape) == tuple(
        decoder_output_shape
    ), f"Input shape {input_shape} != decoder output shape {decoder_output_shape}"

    return encoder, encoder_output_shape, decoder
