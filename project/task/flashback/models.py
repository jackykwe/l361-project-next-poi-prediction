"""RNN model architecture, training, and testing functions for Flashback-MCMG."""

import logging
from enum import Enum

import numpy as np
import torch
from flwr.common.logger import log
from torch import Tensor, nn

from project.types.common import IsolatedRNG

"""
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
"""


# * Adapted from Flashback's code
def create_h0_strategy(seed: int, hidden_size: int, is_lstm: bool):
    if is_lstm:
        return LstmStrategy(
            hidden_size, FixNoiseStrategy(seed, hidden_size), FixNoiseStrategy(seed, hidden_size)
        )
    else:
        return FixNoiseStrategy(seed, hidden_size)


# * Adapted from Flashback's code
class H0Strategy:

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


# * Adapted from Flashback's code
class FixNoiseStrategy(H0Strategy):
    """use fixed normal noise as initialization"""

    def __init__(self, seed: int, hidden_size: int) -> None:
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = (
            torch.randn(
                self.hidden_size,
                generator=torch.Generator().manual_seed(seed),
                requires_grad=False,
            )
            * sd
            + mu
        )

    def on_init(self, user_len: int, device: torch.device) -> Tensor:
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user: int) -> Tensor:
        return self.h0


# * Adapted from Flashback's code
# ! NB. Not supported in this project
class LstmStrategy(H0Strategy):
    """creates h0 and c0 using the inner strategy"""

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        raise NotImplementedError
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h, c)

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h, c)


# * Adapted from Flashback's code
class Rnn(Enum):
    """The available RNN units"""

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == "rnn":
            return Rnn.RNN
        if name == "gru":
            return Rnn.GRU
        if name == "lstm":
            return Rnn.LSTM
        raise ValueError(f"{name} not supported in --rnn")


# * Adapted from Flashback's code
class RnnFactory:
    """Creates the desired RNN unit."""

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return "Use pytorch RNN implementation."
        if self.rnn_type == Rnn.GRU:
            return "Use pytorch GRU implementation."
        if self.rnn_type == Rnn.LSTM:
            return "Use pytorch LSTM implementation."

    def is_lstm(self):
        return self.rnn_type == Rnn.LSTM

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


# * Adapted from Flashback's code
class Flashback(nn.Module):
    """Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    # * input_size is loc_count, the number of locations in total across all users
    def __init__(
        self,
        input_size: int,  # * i.e. loc_count
        user_count: int,
        hidden_size: int,
        lambda_t: float,
        lambda_s: float,
        h0_seed: int,  #* h0 is a constant always
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        # * This is eqn2; delta_t is in units seconds
        # function for computing temporal weight
        self.f_t = lambda delta_t, user_len: (
            (torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2
        ) * torch.exp(
            -(delta_t / 86400 * lambda_t)
        )  # hover cosine + exp decay
        # * This is eqn3; delta_s is in units GPS coordinates
        # function for computing spatial weight
        self.f_s = lambda delta_s, user_len: torch.exp(
            -(delta_s * lambda_s)
        )  # exp decay

        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.rnn = RnnFactory("rnn").create(
            hidden_size
        )  # * seq2seq  (many-to-many RNN)
        self.fc = nn.Linear(
            2 * hidden_size, input_size
        )  # create outputs in lenght of locations  #* many-to-one RNN
        # ! but actually by the way forward() works, it is many-to-many (see shape of the returned y_linear object: one prediction is made for each leading substring of sequence_length==20)
        # * self.fc turns a (2*hidden_size==20) vector into a (loc_count==total number of locations) vector (can be large!)
        # TODO NB: Can never predict never-before-seen locations in the train dataset.

        self.h0_strategy: FixNoiseStrategy = create_h0_strategy(
            h0_seed,
            hidden_size,
            False
        )  # * initial hidden state to RNN

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        s: Tensor,
        y_t: Tensor,
        y_s: Tensor,
        h: Tensor,
        active_user: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # * for shapes of x, t, s, y, y_t, y_s, active_users, see bottom of dataset.py
        # * note that squeeze has been called, so there isn't a prepended batch dimension. The
        # * dimensions stated in the bottom of dataset.py apply here exactly.
        # *! NB: if this is called from trainer.evaluate(), then active_user has NOT been squeezed.
        # *!     if this is called from trainer.loss(), then active_user has been squeezed.
        # *!     It doesn't matter.

        seq_len, user_len = (
            x.size()
        )  # * user_len is the batch_size so yes user_len (see near bottom of dataset.py)
        # * x is of shape (sequence_length==20, batch_size). Loc IDs.
        x_emb = self.encoder(
            x
        )  # * shape is (sequence_length==20, batch_size, hidden_dim==10)
        out, h = self.rnn(x_emb, h)  # * h is of shape (1, batch_size, hidden_dim==10)
        # * out is of shape (sequence_length==20, batch_size, hidden_dim==10)  [the upwards arrows out of an RNN unrolled-view]
        # * h is of shape (1, batch_size, hidden_dim==10)

        # comopute weights per user  #* these are the spatio-temporal weights MULTIPLIED BY the hidden states, hence the self.hidden_size dimension
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        for i in range(seq_len):
            sum_w = torch.zeros(
                user_len, 1, device=x.device
            )  # * shape is (batch_size, 1)
            for j in range(i + 1):
                dist_t = t[i] - t[j]  # * shape is (batch_size,)
                dist_s = torch.norm(s[i] - s[j], dim=-1)  # * shape is (batch_size,)
                a_j = self.f_t(
                    dist_t, user_len
                )  # * second arg not used; shape is (batch_size,)
                b_j = self.f_s(
                    dist_s, user_len
                )  # * second arg not used; shape is (batch_size,)
                a_j = a_j.unsqueeze(1)  # * shape is (batch_size, 1)
                b_j = b_j.unsqueeze(1)  # * shape is (batch_size, 1)
                w_j = (
                    a_j * b_j + 1e-10
                )  # small epsilon to avoid 0 division  #* shape is (batch_size, 1)
                sum_w += w_j
                out_w[i] += (
                    w_j * out[j]
                )  # * out[j] shape is (batch_size, hidden_dim==10)
            # normliaze according to weights
            out_w[
                i
            ] /= sum_w  # * sum across ALL O(n^2) weights associated with deltas (time and space)

        # add user embedding:
        p_u = self.user_encoder(
            active_user
        )  # * shape is (1, batch_size, hidden_size==10); active_user shape is (1, batch_size)
        p_u = p_u.view(
            user_len, self.hidden_size
        )  # * shape is (batch_size, hidden_size==10)
        out_pu = torch.zeros(seq_len, user_len, 2 * self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)
        y_linear = self.fc(
            out_pu
        )  # * (sequence_length==20, batch_size, 2*hidden_size==20) |-> (sequence_length==20, batch_size, loc_count==total number of locations)
        return y_linear, h


# * Custom wrapper to match the NetGenerator Interface
def get_net(
    _config: dict,  # * this is net_config_initial_parameters, then net_config from the yaml
    _rng_tuple: IsolatedRNG,
) -> nn.Module:
    log(
        logging.WARNING,
        f"get_net() using debug_origin={_config['debug_origin']}, _config={_config}",
    )
    return Flashback(
        h0_seed=_config["h0_seed"],
        input_size=_config["loc_count"],
        user_count=_config["user_count"],
        hidden_size=_config["hidden_dim"],
        lambda_t=_config["lambda_t"],
        lambda_s=_config["lambda_s"],
    )
