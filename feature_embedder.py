from typing import Optional

import torch
import numpy as np
import math


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class FeatureEmbedder(torch.nn.Module):
    """
    Combines each feature value with its feature ID. The embedding of feature IDs is a trainable parameter.

    This is analogous to position encoding in a transformer.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        device: torch.device=torch.device("cpu"),
        metadata: Optional[torch.Tensor]=None,
        multiply_weights: bool = False,
    ):
        """
        Args:
            input_dim (int): Number of features.
            embedding_dim (int): Size of embedding for each feature ID.
            metadata (Optional[torch.Tensor]): Each row represents a feature and each column is a metadata dimension for the feature.
                Shape (input_dim, metadata_dim).
            device (torch.device): Pytorch device to use.
            multiply_weights (bool): Whether or not to take the product of x with embedding weights when feeding through.
        """
        super().__init__()
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        if metadata is not None:
            assert metadata.shape[0] == input_dim
        self._metadata = metadata
        self._multiply_weights = multiply_weights

        # ravi self._embedding_weights = torch.nn.Parameter(
        #     torch.zeros(input_dim, embedding_dim, device=device), requires_grad=True
        # )

        # ravi self._embedding_bias = torch.nn.Parameter(torch.zeros(input_dim, 1, device=device), requires_grad=True)
        self.zeros = torch.zeros(input_dim, 1)
        # torch.nn.init.xavier_uniform_(self._embedding_weights)
        # torch.nn.init.xavier_uniform_(self._embedding_bias)

    @property
    def output_dim(self) -> int:
        """
        The final output dimension depends on how features and embeddings are combined in the forward method.

        Returns:
            output_dim (int): The final output dimension of the feature embedder.
        """
        metadata_dim = 0 if self._metadata is None else self._metadata.shape[1]
        output_dim = metadata_dim + self._embedding_dim + 2
        return output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Map each element of each set to a vector, according to which feature it represents.

        Args:
            x (torch.Tensor): Data to embed of shape (batch_size, input_dim).

        Returns:
            feature_embedded_x (torch.Tensor): of shape (batch_size * input_dim, output_dim)
        """
        # cat, val, cm, vm = x
        batch_size, seq_len, feat_dim = x.shape  # Shape (batch_size, input_dim).
        # cat_flat = cat#.reshape(batch_size * self._input_dim, 1)
        # val_flat = val#.reshape(batch_size * self._input_dim, 1)
        # cm_flat = cm#.reshape(batch_size * self._input_dim, 1)
        # vm_flat = vm#.reshape(batch_size * self._input_dim, 1)

        self._embedding_weights = positionalencoding1d(feat_dim, seq_len)
        # Repeat weights and bias for each instance of each feature.
        if self._metadata is not None:
            embedding_weights_and_metadata = torch.cat((self._embedding_weights, self._metadata), dim=1)
            repeat_embedding_weights = embedding_weights_and_metadata.repeat([batch_size, 1, 1])
        else:
            repeat_embedding_weights = self._embedding_weights.repeat([batch_size, 1, 1])

        # Shape (batch_size * input_dim, embedding_dim)
        #repeat_embedding_weights = repeat_embedding_weights.reshape([batch_size * self._input_dim, -1]).to(x.device)
        repeat_embedding_weights = repeat_embedding_weights.to(x.device)

        # ravi
        # repeat_embedding_bias = self._embedding_bias.repeat((batch_size, 1, 1))
        # repeat_embedding_bias = repeat_embedding_bias.reshape((batch_size * self._input_dim, 1))

        # X_flat = x.reshape((batch_size*self._input_dim), -1)

        # if self._multiply_weights:
        #     pass
        #     # features_to_concatenate = [
        #     #     x_flat,
        #     #     x_flat * repeat_embedding_weights,
        #     #     repeat_embedding_bias,
        #     # ]
        # else:
        #     features_to_concatenate = [
        #         X_flat,
        #         repeat_embedding_weights,
        #         #repeat_embedding_bias, ravi
        #     ]

        # # Shape (batch_size*input_dim, output_dim)
        # feature_embedded_x = torch.cat(features_to_concatenate, dim=1)
        # feature_embedded_x = feature_embedded_x.reshape((batch_size, self._input_dim, -1))
        op = x+repeat_embedding_weights
        return op #feature_embedded_x

    def __repr__(self):
        return f"FeatureEmbedder(input_dim={self._input_dim}, embedding_dim={self._embedding_dim}, multiply_weights={self._multiply_weights}, output_dim={self.output_dim})"


class SparseFeatureEmbedder(FeatureEmbedder):
    """
    Feature embedder to use with SparsePointNet.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # type: ignore
        """

        Maps each observed element of each set to a vector, according to which feature it represents.

        Args:
            x (torch.Tensor): Data to embed of shape (batch_size, input_dim).
            mask (torch.Tensor): Mask of shape (batch_size, input_dim) indicating observed variables.
                1 is observed, 0 is un-observed.

        Returns:
            feature_embedded_x (torch.Tensor):
                Observed features, embedded with their feature IDs of shape (num_observed_features, output_dim).

        """
        if self._metadata is not None:
            raise NotImplementedError("metadata parameter is not currently supported in SparseFeatureEmbedder.")

        batch_size, _ = x.size()

        # Reshape input so that the first dimension contains all input datapoints stacked on top of one another
        x_flat = x.reshape(batch_size * self._input_dim, 1)
        mask = mask.reshape(batch_size * self._input_dim, 1)

        # Select only the observed features
        obs_features = torch.nonzero(mask, as_tuple=False)[:, 0]
        x_flat = x_flat[
            obs_features,
        ]  # shape (num_observed_features, 1)

        # Repeat weights and bias for each observed instance of each feature.
        repeat_embedding_weights = self._embedding_weights.repeat([batch_size, 1, 1])
        repeat_embedding_weights = repeat_embedding_weights.reshape([batch_size * self._input_dim, self._embedding_dim])

        # shape (num_observed_features, embedding_dim)
        repeat_embedding_weights = repeat_embedding_weights[
            obs_features,
        ]

        repeat_embedding_bias = self._embedding_bias.repeat((batch_size, 1, 1))
        repeat_embedding_bias = repeat_embedding_bias.reshape((batch_size * self._input_dim, 1))

        # shape (num_observed_features, 1)
        repeat_embedding_bias = repeat_embedding_bias[
            obs_features,
        ]

        if self._multiply_weights:
            features_to_concatenate = [
                x_flat,
                x_flat * repeat_embedding_weights,
                repeat_embedding_bias,
            ]
        else:
            features_to_concatenate = [
                x_flat,
                repeat_embedding_weights,
                repeat_embedding_bias,
            ]

        # Shape (num_observed_features, output_dim)
        return torch.cat(features_to_concatenate, dim=1)
