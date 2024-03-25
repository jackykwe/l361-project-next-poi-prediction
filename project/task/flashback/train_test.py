"""Flashback-MCMG training and testing functions, local and federated."""

import logging
from collections.abc import Sized
from pathlib import Path
from typing import cast

import numpy as np
import torch
from flwr.common.logger import log
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)
from project.task.flashback.dataset import PoiDataset
from project.task.flashback.models import Flashback
from project.types.common import IsolatedRNG


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    # * See client.py line 129. God knows why the framework is written like this; since
    # * there is a mix of both things from the yaml and things from within code (e.g.
    # * client.py line 129), this does NOT serve as a schema for what needs to be in the
    # * yaml. This is infinitely confusing.
    device: torch.device
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    loc_count: int
    client_dropout_probability: float
    server_round: int

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(  # pylint: disable=too-many-arguments
    net: Flashback,  # * used task.fit_config.net_config
    trainloader: DataLoader,  # * used task.fit_config.dataloader_config
    _config: dict,  # * this is task.fit_config.run_config, which is described by TrainConfig above (??? why the difference in name lmao) WITHOUT the device field.
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    log(logging.WARNING, f"train() called with _config={_config}")
    config: TrainConfig = TrainConfig(**_config)
    del _config

    if _rng_tuple[1].random() <= config.client_dropout_probability:
        # Client dropped out; do not consider this client in the round's training
        return 1, {"train_loss": 0}

    net.to(config.device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40, 60, 80], gamma=0.2
    )
    for _ in range((config.server_round - 1) * config.epochs):
        scheduler.step()

    with logging_redirect_tqdm():
        for e in tqdm(range(config.epochs), desc="Epoch"):
            h = net.h0_strategy.on_init(
                config.batch_size, config.device
            )  # * hidden states are preserved across dataloader iterations for as long as each user's sequence lasts. Shape is (1, setting.batch_size, setting.hidden_dim).
            cast(
                PoiDataset, trainloader.dataset
            ).shuffle_users()  # shuffle users before each epoch!

            losses = []

            # *     x.shape is (sequence_length==20, batch_size)
            # *     t.shape is (sequence_length==20, batch_size)
            # *     s.shape is (sequence_length==20, batch_size, 2)
            # *     y.shape is (sequence_length==20, batch_size)
            # *     y_t.shape is (sequence_length==20, batch_size)
            # *     y_s.shape is (sequence_length==20, batch_size, 2)
            # *     len(reset_h) is batch_size
            # *     active_users.shape is (batch_size,); a tensor of the user IDs corresponding to each batch (in the batch_size dimension; each batch corresponds to one user ID)
            for x, t, s, y, y_t, y_s, reset_h, active_users in tqdm(
                trainloader, desc="Dataloader (train)", leave=False
            ):
                # reset hidden states for newly added users
                for j, reset in enumerate(reset_h):
                    if reset:
                        # if setting.is_lstm:
                        #     hc = h0_strategy.on_reset(active_users[0][j])
                        #     h[0][0, j] = hc[0]
                        #     h[1][0, j] = hc[1]
                        # else:
                        h[0, j] = net.h0_strategy.on_reset(active_users[0][j])

                # * Need to squeeze: dataloader prepends the batch dimension, which is 1
                x = x.squeeze(dim=0).to(config.device)
                t = t.squeeze(dim=0).to(config.device)
                s = s.squeeze(dim=0).to(config.device)
                y = y.squeeze(dim=0).to(config.device)
                y_t = y_t.squeeze(dim=0).to(config.device)
                y_s = y_s.squeeze(dim=0).to(config.device)
                active_users = active_users.to(config.device)

                optimizer.zero_grad()

                """ takes a batch (users x location sequence)
                and corresponding targets in order to compute the training loss """
                # * for shapes of x, t, s, y, y_t, y_s, reset_h, active_users, see bottom of PoiDataset.__getitem__()
                # * note that squeeze has been called, so there isn't a prepended batch dimension. The
                # * dimensions stated in the bottom of PoiDataset.__getitem__() apply here exactly.
                # *! NB: active_user here has NOT been squeezed, different from evaluate().
                # *!     It doesn't matter.

                out, h = net(x, t, s, y_t, y_s, h, active_users)
                # * out shape is (sequence_length==20, batch_size, loc_count==total number of locations)
                # * h shape (1, batch_size, hidden_dim==10)

                h = (
                    h.detach()
                )  # * fix courtesy of https://github.com/eXascaleInfolab/Flashback_code/issues/2#issuecomment-1149924973
                out = out.view(
                    -1, config.loc_count
                )  # * shape is  (sequence_length==20 * batch_size, loc_count==total number of locations)
                y = y.view(-1)  # * flatten y into a single dimension vector;
                # * (sequence_length==20, batch_size) |-> (sequence_length==20 * batch_size,)
                loss = criterion(out, y)
                loss.backward(retain_graph=True)
                losses.append(loss.item())
                optimizer.step()
                # * See https://stackoverflow.com/a/53975741 for a discussion on how loss.backward() and
                # * optimizer.step() are linked: essentially the gradients are stored on the model parameter
                # * tensors themselves; model parameter tensors have an implicit computational graph, and when
                # * loss takes in the predicted label (and compares it with the actual label), gradients are
                # * calculated from predicted label backwards through its computational graph. These gradients
                # * are stored on the tensors involved in the computational graph. optimizer.step() then iterates
                # * through model parameters and uses the stored gradients to update them.

            scheduler.step()

            if (e + 1) % 1 == 0:
                epoch_loss = np.mean(losses)
                log(logging.INFO, f"Epoch: {e + 1}/{config.epochs}")
                log(logging.INFO, f"Used learning rate: {scheduler.get_lr()[0]}")
                log(logging.INFO, f"Avg Loss: {epoch_loss}")

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": epoch_loss,  # * average loss per sample for the last epoch
    }


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    # * See client.py line 129. God knows why the framework is written like this; since
    # * there is a mix of both things from the yaml and things from within code (e.g.
    # * client.py line 129), this does NOT serve as a schema for what needs to be in the
    # * yaml. This is infinitely confusing.
    device: torch.device
    batch_size: int
    # epochs: int
    # learning_rate: float
    # weight_decay: float
    # loc_count: int

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: Flashback,  # * used task.eval_config.net_config
    testloader: DataLoader,  # * used task.eval_config.dataloader_config
    _config: dict,  # * this is task.eval_config.run_config, which is described by TrainConfig above (??? why the difference in name lmao) WITHOUT the device field.
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    _config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior


    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    log(logging.WARNING, f"test() called with _config={_config}")
    config: TestConfig = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    cast(PoiDataset, testloader.dataset).reset()
    h = net.h0_strategy.on_init(config.batch_size, config.device)

    criterion = nn.CrossEntropyLoss()
    # correct, per_sample_loss = 0, 0.0

    with torch.no_grad():
        iter_cnt = 0
        recall1 = 0
        recall5 = 0
        recall10 = 0
        average_precision = 0.0

        u_iter_cnt = np.zeros(net.user_count)
        u_recall1 = np.zeros(net.user_count)
        u_recall5 = np.zeros(net.user_count)
        u_recall10 = np.zeros(net.user_count)
        u_average_precision = np.zeros(net.user_count)
        reset_count = torch.zeros(net.user_count)

        losses = []

        # * for shapes of x, t, s, y, y_t, y_s, reset_h, active_users, see bottom of PoiDataset.__getitem__()
        # * note that until we call squeeze, there is a prepended batch dimension
        # *     x.shape is (sequence_length==20, batch_size)
        # *     t.shape is (sequence_length==20, batch_size)
        # *     s.shape is (sequence_length==20, batch_size, 2)
        # *     y.shape is (sequence_length==20, batch_size)
        # *     y_t.shape is (sequence_length==20, batch_size)
        # *     y_s.shape is (sequence_length==20, batch_size, 2)
        # *     len(reset_h) is batch_size
        # *     active_users.shape is (batch_size,); a tensor of the user IDs corresponding to each batch (in the batch_size dimension; each batch corresponds to one user ID)
        with logging_redirect_tqdm():
            for x, t, s, y, y_t, y_s, reset_h, active_users in tqdm(
                testloader, desc="Dataloader (test)"
            ):
                active_users = active_users.squeeze(dim=0)
                for j, reset in enumerate(reset_h):
                    if reset:
                        # if self.setting.is_lstm:
                        #     hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        #     h[0][0, j] = hc[0]
                        #     h[1][0, j] = hc[1]
                        # else:
                        h[0, j] = net.h0_strategy.on_reset_test(
                            active_users[j], config.device
                        )
                        reset_count[active_users[j]] += 1

                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze(dim=0).to(config.device)
                t = t.squeeze(dim=0).to(config.device)
                s = s.squeeze(dim=0).to(config.device)
                y = y.squeeze(dim=0)
                y_t = y_t.squeeze(dim=0).to(config.device)
                y_s = y_s.squeeze(dim=0).to(config.device)

                active_users = active_users.to(config.device)

                # evaluate:
                """ takes a batch (users x location sequence)
                then does the prediction and returns a list of user x sequence x location
                describing the probabilities for each location at each position in the sequence.
                t, s are temporal and spatial data related to the location sequence x
                y_t, y_s are temporal and spatial data related to the target sequence y.
                Flashback does not access y_t and y_s for prediction!
                """
                # * for shapes of x, t, s, y, y_t, y_s, reset_h, active_users, see bottom of PoiDataset.__getitem__()
                # * note that squeeze has been called, so there isn't a prepended batch dimension. The
                # * dimensions stated in the bottom of PoiDataset.__getitem__() apply here exactly.
                # *! NB: active_user here has been squeezed, different from loss().
                # *!     It doesn't matter.

                out, h = net(x, t, s, y_t, y_s, h, active_users)
                # * out shape is (sequence_length==20, batch_size, loc_count==total number of locations)
                # * h shape (1, batch_size, hidden_dim==10)

                # * Frankenstein'ed code; to calculate loss
                # * shape is  (sequence_length==20 * batch_size, loc_count==total number of locations)
                losscalc_out = out.view(-1, net.input_size)
                losscalc_y = y.view(-1)  # * flatten y into a single dimension vector;
                # * (sequence_length==20, batch_size) |-> (sequence_length==20 * batch_size,)
                losscalc_loss = criterion(losscalc_out, losscalc_y)
                losses.append(losscalc_loss.item())

                out = out.transpose(0, 1)  # ! model outputs logits
                # * out shape is (batch_size, sequence_length==20, loc_count==total number of locations)
                # * h shape is (1, batch_size, hidden_dim==10)

                for j in range(config.batch_size):
                    # o contains a per user list of votes for all locations for each sequence entry
                    # * shape is (sequence_length==20, loc_count==total number of locations)
                    o = out[j]

                    # partition elements
                    # * shape is (sequence_length==20, loc_count==total number of locations)
                    o_n = o.cpu().detach().numpy()
                    # * shape is (sequence_length==20, top 10 location IDs)
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:]  # top 10 elements

                    # * shape is (sequence_length==20,);  y shape is (sequence_length==20, batch_size)
                    y_j = y[:, j]

                    for k in range(len(y_j)):
                        if reset_count[active_users[j]] > 1:
                            continue  # skip already evaluated users.

                        # resort indices for k:
                        ind_k = ind[k]  # * shape is (top 10 location IDs,)
                        # sort top 10 elements descending
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]

                        # * shape is (top 10 location IDs in descending order,)
                        r = torch.tensor(r)
                        t = y_j[k]  # * shape is (1,), the correct answer

                        # compute MAP:  #* mean average precision
                        # * shape is (loc_count==total number of locations,)
                        r_kj = o_n[k, :]
                        # * shape is (1,), the predicted "probability" (? no softmax but ok) for the correct answer
                        t_val = r_kj[t]
                        upper = np.where(r_kj > t_val)[
                            0
                        ]  # * [0] to obtain from a 1-tuple
                        precision = 1.0 / (1 + len(upper))

                        # store
                        u_iter_cnt[active_users[j]] += 1
                        u_recall1[active_users[j]] += t in r[:1]
                        u_recall5[active_users[j]] += t in r[:5]
                        u_recall10[active_users[j]] += t in r[:10]
                        # ! TODO This precision measure is wack... No. It's similar to NDCG.
                        u_average_precision[active_users[j]] += precision

        formatter = "{0:.8f}"
        for j in range(net.user_count):
            iter_cnt += u_iter_cnt[j]
            recall1 += u_recall1[j]
            recall5 += u_recall5[j]
            recall10 += u_recall10[j]
            average_precision += u_average_precision[j]

            # if (self.setting.report_user > 0 and (j+1) % self.setting.report_user == 0):
            log(
                logging.INFO,
                "\t".join((
                    "Report user",
                    str(j),
                    "preds:",
                    str(u_iter_cnt[j]),
                    "recall@1",
                    formatter.format(u_recall1[j] / u_iter_cnt[j]),
                    "MAP",
                    formatter.format(u_average_precision[j] / u_iter_cnt[j]),
                )),
            )

    log(logging.INFO, f"recall@1: {formatter.format(recall1 / iter_cnt)}")
    log(logging.INFO, f"recall@5: {formatter.format(recall5 / iter_cnt)}")
    log(logging.INFO, f"recall@10: {formatter.format(recall10 / iter_cnt)}")
    log(logging.INFO, f"MAP: {formatter.format(average_precision / iter_cnt)}")
    log(logging.INFO, f"predictions: {iter_cnt}")

    return (
        np.mean(losses),
        len(cast(Sized, testloader.dataset)),
        {
            "recall@1": recall1 / iter_cnt,
            "recall@5": recall5 / iter_cnt,
            "recall@10": recall10 / iter_cnt,
            "MAP": average_precision / iter_cnt,
            "predictions": iter_cnt,
        },
    )


# Use defaults as they are completely determined
# by the other functions defined in mnist_classification

# * main.py: "This is the function that is called on the server to evaluated the global model"
# * lab1p2: "Function to test the federated model performance external to a client instantiation"
get_fed_eval_fn = get_default_fed_eval_fn

get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
