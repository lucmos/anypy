import logging
from typing import Optional, Sequence, Tuple

import hydra.utils
import torch
from torch import nn

pylogger = logging.getLogger(__name__)


def infer_dimension(width: int, height: int, n_channels: int, model: nn.Module, batch_size: int = 8) -> torch.Tensor:
    """Compute the output of a model given a fake batch.

    Args:
        width: the width of the image to generate the fake batch
        height:  the height of the image to generate the fake batch
        n_channels:  the n_channels of the image to generate the fake batch
        model: the model to use to compute the output
        batch_size: batch size to use for the fake output

    Returns:
        the fake output
    """
    with torch.no_grad():
        model.eval()
        fake_batch = torch.zeros([batch_size, n_channels, width, height])
        fake_out = model(fake_batch)
        model.train()
        return fake_out


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


def build_transposed_convolution(
    in_channels: int,
    out_channels: int,
    target_output_width,
    target_output_height,
    input_width,
    input_height,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (1, 1),
    output_padding: Tuple[int, int] = (0, 0),
    dilation: int = 1,
) -> nn.ConvTranspose2d:
    # kernel_w = (metadata.width - (fake_out.width −1)×stride[0] + 2×padding[0] - output_padding[0]  - 1)/dilation[0] + 1
    # kernel_h = (metadata.height - (fake_out.height −1)×stride[1] + 2×padding[1] - output_padding[1]  - 1)/dilation[1] + 1
    kernel_w = (
        target_output_width - (input_width - 1) * stride[0] + 2 * padding[0] - output_padding[0] - 1
    ) / dilation + 1
    kernel_h = (
        target_output_height - (input_height - 1) * stride[1] + 2 * padding[1] - output_padding[1] - 1
    ) / dilation + 1
    assert kernel_w > 0 and kernel_h > 0

    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(int(kernel_w), int(kernel_h)),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )


def build_dynamic_encoder_decoder(
    width,
    height,
    n_channels,
    hidden_dims: Optional[Sequence[int]],
    activation: str = "torch.nn.GELU",
    use_batch_norm: bool = True,
    remove_encoder_last_activation: bool = False,
) -> Tuple[nn.Module, Sequence[int], nn.Module]:
    """Builds a dynamic convolutional encoder-decoder pair with parametrized hidden dimensions number and size.

    Args:
        width: the width of the images to work with
        height: the height of the images
        n_channels: the number of channels of the images
        hidden_dims: a sequence of ints to specify the number and size of the hidden layers in the encoder and decoder
        activation: the activation function to use
        use_batch_norm: whether to use batch normalization
        remove_encoder_last_activation: whether to remove the last activation in the encoder

    Returns:
        the encoder model, the shape in the latent space assuming batch size 1, the decoder module.
    """
    modules = []

    if hidden_dims is None:
        hidden_dims = (32, 64, 128, 256)

    STRIDE = (2, 2)
    PADDING = (1, 1)

    # Build Encoder
    encoder_shape_sequence = [
        [width, height],
    ]
    running_channels = n_channels
    for i, h_dim in enumerate(hidden_dims):
        modules.append(
            nn.Sequential(
                (
                    conv2d := nn.Conv2d(
                        running_channels, out_channels=h_dim, kernel_size=4, stride=STRIDE, padding=PADDING
                    )
                ),
                nn.BatchNorm2d(h_dim) if use_batch_norm else nn.Identity(),
                nn.Identity()
                if i == len(hidden_dims) - 1 and remove_encoder_last_activation
                else hydra.utils.instantiate({"_target_": activation}),
            )
        )
        conv2d_out = infer_dimension(
            encoder_shape_sequence[-1][0],
            encoder_shape_sequence[-1][1],
            running_channels,
            conv2d,
        )
        encoder_shape_sequence.append([conv2d_out.shape[2], conv2d_out.shape[3]])
        running_channels = h_dim

    encoder = nn.Sequential(*modules)

    encoder_out_shape = infer_dimension(width, height, n_channels=n_channels, model=encoder, batch_size=1).shape

    pylogger.info(f"Encoder output shape: {encoder_out_shape}")

    # Build Decoder
    hidden_dims = list(reversed(hidden_dims))
    hidden_dims = hidden_dims + hidden_dims[-1:]

    running_input_width = encoder_out_shape[2]
    running_input_height = encoder_out_shape[3]
    modules = []
    for i, (target_output_width, target_output_height) in zip(
        range(len(hidden_dims) - 1), reversed(encoder_shape_sequence[:-1])
    ):
        modules.append(
            nn.Sequential(
                build_transposed_convolution(
                    in_channels=hidden_dims[i],
                    out_channels=hidden_dims[i + 1],
                    target_output_width=target_output_width,
                    target_output_height=target_output_height,
                    input_width=running_input_width,
                    input_height=running_input_height,
                    stride=STRIDE,
                    padding=PADDING,
                ),
                nn.BatchNorm2d(hidden_dims[i + 1]) if use_batch_norm else nn.Identity(),
                hydra.utils.instantiate({"_target_": activation}),
            )
        )
        running_input_width = target_output_width
        running_input_height = target_output_height

    decoder = nn.Sequential(
        *modules,
        nn.Sequential(
            nn.Conv2d(hidden_dims[-1], out_channels=n_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ),
    )
    return encoder, encoder_out_shape, decoder
