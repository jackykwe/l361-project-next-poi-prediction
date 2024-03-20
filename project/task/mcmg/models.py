"""CNN model architecture, training, and testing functions for MNIST."""
from typing import Callable, Any

import torch
from torch import nn

from project.task.mcmg.mcmg_model import Model
from project.types.common import IsolatedRNG


class MCMG(nn.Module):

    def __init__(self, config, rng) -> None:
        """Create the MCMG model wrapper.

        Parameters
        ----------
        num_classes : int
            Number of classes in the dataset.

        Returns
        -------
        None
        """

        super().__init__()
        self.epoch = config["epoch"]
        self.model = Model(config["hidden_size"], config["lr"], config["l2"], config["step"], config["n_head"],
                           config["k_blocks"],
                           config, config["POI_n_node"], config["cate_n_node"], config["regi_n_node"],
                           config["time_n_node"],
                           config["POI_dist_n_node"], config["regi_dist_n_node"], config["len_max"])

    def forward(
            self, POI_inputs, POI_A, cate_inputs, regi_inputs, time_inputs, POI_dist_inputs, regi_dist_inputs
    ) -> torch.Tensor:
        """Forward pass of the model."""

        return self.model.forward(POI_inputs, POI_A, cate_inputs, regi_inputs, time_inputs, POI_dist_inputs,
                                  regi_dist_inputs)


def get_mcmg_model(config, rng_tuple) -> Callable[[dict, IsolatedRNG], Any]:
    return MCMG(config, rng_tuple)
