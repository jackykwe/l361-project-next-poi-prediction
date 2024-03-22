"""Flashback-MCMG dataset utilities for federated learning."""

import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from random import shuffle
from typing import Any, cast

import torch
from flwr.common.logger import log
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader

from project.types.common import CID, ClientDataloaderGen, FedDataloaderGen, IsolatedRNG


# * Used for verifying Hydra configs
class ClientDataloaderConfig(BaseModel):
    """Dataloader configuration for the client.

    Allows '.' member access and static checking. Guarantees that all necessary
    components are present, fails early if config is mismatched to dataloader.
    """

    loc_count: int
    batch_size: int
    # sequence_length: just here for a secondary check that the dataset_preparation was
    # successful; Flashback was coded like this so I'll respect it
    sequence_length: int
    city: str  # used to select dataset
    partition_type: str  # used to select between centralised/federated setting

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


class FedDataloaderConfig(BaseModel):
    """Dataloader configuration for the client.

    Allows '.' member access and static checking. Guarantees that all necessary
    components are present, fails early if config is mismatched to dataloader.
    """

    loc_count: int
    batch_size: int
    # sequence_length: just here for a secondary check that the dataset_preparation was
    # successful; Flashback was coded like this so I'll respect it
    sequence_length: int
    city: str  # used to select dataset
    partition_type: str  # used to select between centralised/federated setting

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


# * Adapted from our Flashback fork, https://github.com/jackykwe/Flashback_code
class Split(Enum):
    """Defines whether to split for train or test."""

    TRAIN = 0
    TEST = 1


# * Adapted from our Flashback fork, https://github.com/jackykwe/Flashback_code
class Usage(Enum):
    """
    Each user has a different amount of sequences. The usage defines
    how many sequences are used:

    MAX: each sequence of any user is used (default)
    MIN: only as many as the minimal user has
    CUSTOM: up to a fixed amount if available.

    The unused sequences are discarded. This setting applies after the train/test split.
    """

    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2


# * Adapted from our Flashback fork, https://github.com/jackykwe/Flashback_code
# * I have manually stepped through the entirety of Flashback's code, and everything
# * looks sensible
class PoiDataset(torch.utils.data.Dataset):
    """
    Our Point-of-interest pytorch dataset: To maximize GPU workload we organize the data
    in batches of "user" x "a fixed length sequence of locations". The active users have
    at least one sequence in the batch.
    In order to fill the batch all the time we wrap around the available users: if an
    active user runs out of locations we replace him with a new one. When there are no
    unused users available we reuse already processed ones. This happens if a single
    user was way more active than the average user. The batch guarantees that each
    sequence of each user was processed at least once.

    This data management has the implication that some sequences might be processed
    twice (or more) per epoch.
    During trainig you should call PoiDataset::shuffle_users before the start of a new
    epoch. This leads to more stochastic as different sequences will be processed twice.
    During testing you *have to* keep track of the already processed users.

    Working with a fixed sequence length omits awkward code by removing only few of the
    latest checkins per user.
    We work with a 80/20 train/test spilt, where test check-ins are strictly after
    training checkins.    To obtain at least one test sequence with label we require any
    user to have at least (5*<sequence-length>+1) checkins in total.
    """

    def reset(self) -> None:
        # reset training state:
        self.next_user_idx: int = 0  # current user index to add
        self.active_users: list[int] = []  # current active users
        self.active_user_seq: list[int] = []  # current active users sequences
        self.user_permutation: list[int] = []  # shuffle users during training

        # set active users:
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(i)
            self.active_user_seq.append(0)

        # use 1:1 permutation:
        for i in range(len(self.users)):
            self.user_permutation.append(i)

    def shuffle_users(self) -> None:
        shuffle(self.user_permutation)
        # reset active users:
        # * this is a pointer into the self.user_permutation (reshuffled in each epoch)
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(self.user_permutation[i])
            self.active_user_seq.append(0)

    def __init__(
        self,
        users: list[int],
        times: list[list[float]],
        coords: list[list[tuple[float, float]]],
        locs: list[list[int]],
        sequence_length: int,
        batch_size: int,
        split: Split,
        usage: Usage,
        loc_count: int,
        custom_seq_count: int,
    ) -> None:
        # * Inputs:
        # * times: list of (one list per user), from t=0 to t=T-1
        # * coords: list of (one list per user), from t=0 to t=T-1
        # * locs: list of (one list per user), from t=0 to t=T-1

        self.users: list[int] = users  # * list of (one user ID per user)
        # * list of (one list per user)  # These are the times of Xs, from t=0 to t=T-2
        self.times: list[list[float]] = times
        # * list of (one list per user)  # These are the coords of Xs, from t=0 to t=T-2
        self.coords: list[list[tuple[float, float]]] = coords
        # * list of (one list per user)  # These are the Xs, from t=0 to t=T-2
        self.locs: list[list[int]] = locs
        # * list of (one list per user)
        # * These are the Ys, from t=1 to t=T-1, set as label0 to labelT-2
        self.labels: list[list[int]] = []
        # * list of (one list per user)
        # * These are the times of Ys, from t=1 to t=T-1, set as label0 to labelT-2
        self.lbl_times: list[list[float]] = []
        # * list of (one list per user)
        # * These are the coords of Ys, from t=1 to t=T-1, set as label0 to labelT-2
        self.lbl_coords: list[list[tuple[float, float]]] = []
        # * list of (one list of full--length-20--sequences per user). loc IDs.
        # * MAY SKIP SOME DATA POINTS
        self.sequences: list[list[list[int]]] = []
        # * list of (one list of full--length-20--sequences per user).
        # * MAY SKIP SOME DATA POINTS
        self.sequences_times: list[list[list[float]]] = []
        # * list of (one list of full--length-20--sequences per user).
        # * MAY SKIP SOME DATA POINTS
        self.sequences_coords: list[list[list[tuple[float, float]]]] = []
        # * list of (one list of full--length-20--sequences per user). loc IDs.
        # * MAY SKIP SOME DATA POINTS
        self.sequences_labels: list[list[list[int]]] = []
        # * list of (one list of full--length-20--sequences per user).
        # * MAY SKIP SOME DATA POINTS
        self.sequences_lbl_times: list[list[list[float]]] = []
        # * list of (one list of full--length-20--sequences per user).
        # * MAY SKIP SOME DATA POINTS
        self.sequences_lbl_coords: list[list[list[tuple[float, float]]]] = []
        # * list of (one number of full--length-20--sequences per user)
        self.sequences_count: list[int] = []
        self.Ps: list[Any] = []  # * ???; not used
        self.Qs: Tensor = torch.zeros(loc_count, 1)
        self.usage: Usage = usage
        self.batch_size: int = batch_size
        self.loc_count: int = (
            loc_count  # * number of locations in total across all users
        )
        self.custom_seq_count: int = custom_seq_count  # * ???; not used

        self.reset()

        # collect locations:
        for i in range(loc_count):
            self.Qs[i, 0] = i

        # align labels to locations (shift by one)
        for i, loc in enumerate(locs):
            self.locs[i] = loc[:-1]
            self.labels.append(loc[1:])
            # adapt time and coords:
            self.lbl_times.append(self.times[i][1:])
            self.lbl_coords.append(self.coords[i][1:])
            self.times[i] = self.times[i][:-1]
            self.coords[i] = self.coords[i][:-1]

        # split to training / test phase:  #* this iterates over users i
        for i, (time, coord, loc, label, lbl_time, lbl_coord) in enumerate(
            zip(
                self.times,
                self.coords,
                self.locs,
                self.labels,
                self.lbl_times,
                self.lbl_coords,
                strict=False,
            )
        ):
            train_thr = int(len(loc) * 0.8)
            log(logging.DEBUG, f"i={i} |-> train_thr={train_thr}")  # * DEBUG SET A
            if split == Split.TRAIN:
                self.times[i] = time[:train_thr]
                self.coords[i] = coord[:train_thr]
                self.locs[i] = loc[:train_thr]
                self.labels[i] = label[:train_thr]
                self.lbl_times[i] = lbl_time[:train_thr]
                self.lbl_coords[i] = lbl_coord[:train_thr]
            if split == Split.TEST:
                self.times[i] = time[train_thr:]
                self.coords[i] = coord[train_thr:]
                self.locs[i] = loc[train_thr:]
                self.labels[i] = label[train_thr:]
                self.lbl_times[i] = lbl_time[train_thr:]
                self.lbl_coords[i] = lbl_coord[train_thr:]

        # split location and labels to sequences:
        # * maximum number of full--length-20--sequences found for a particular user
        self.max_seq_count: int = 0
        # * minimum number of full--length-20--sequences found for a particular user
        self.min_seq_count: int = 10_000_000
        # * total number of full--length-20--sequences across all users
        self.capacity: int = 0
        # * this iterates over users i
        for i, (time, coord, loc, label, lbl_time, lbl_coord) in enumerate(
            zip(
                self.times,
                self.coords,
                self.locs,
                self.labels,
                self.lbl_times,
                self.lbl_coords,
                strict=False,
            )
        ):
            # * this is floor; how many full sequence_lengths we have in loc.
            # * The following check asserts that loc has at least sequence_length==20
            # * elements
            seq_count = len(loc) // sequence_length

            # * search "DEBUG SET A" in codebase to debug this
            assert seq_count > 0, (
                f"fix seq-length and min-checkins in order to have "
                f"at least one test sequence in a 80/20 split!; len(loc)={len(loc)}, "
                f"sequence_length={sequence_length}, len(loc)//sequence_length="
                f"{len(loc) // sequence_length}; user ID after mapping={i}"
            )

            seqs: list[list[int]] = []
            seq_times: list[list[float]] = []
            seq_coords: list[list[tuple[float, float]]] = []
            seq_lbls: list[list[int]] = []
            seq_lbl_times: list[list[float]] = []
            seq_lbl_coords: list[list[tuple[float, float]]] = []
            for j in range(seq_count):
                start = j * sequence_length
                end = (j + 1) * sequence_length
                seqs.append(loc[start:end])
                seq_times.append(time[start:end])
                seq_coords.append(coord[start:end])
                seq_lbls.append(label[start:end])
                seq_lbl_times.append(lbl_time[start:end])
                seq_lbl_coords.append(lbl_coord[start:end])
            self.sequences.append(seqs)
            self.sequences_times.append(seq_times)
            self.sequences_coords.append(seq_coords)
            self.sequences_labels.append(seq_lbls)
            self.sequences_lbl_times.append(seq_lbl_times)
            self.sequences_lbl_coords.append(seq_lbl_coords)
            self.sequences_count.append(seq_count)
            self.capacity += seq_count
            self.max_seq_count = max(self.max_seq_count, seq_count)
            self.min_seq_count = min(self.min_seq_count, seq_count)

        # statistics
        if self.usage == Usage.MIN_SEQ_LENGTH:
            log(
                logging.INFO,
                f"{split} load {len(users)} users with min_seq_count {self.min_seq_count} batches: {self.__len__()}",
            )
        if self.usage == Usage.MAX_SEQ_LENGTH:
            log(
                logging.INFO,
                f"{split} load {len(users)} users with max_seq_count {self.max_seq_count} batches: {self.__len__()}",
            )
        if self.usage == Usage.CUSTOM:
            log(
                logging.INFO,
                f"{split} load {len(users)} users with custom_seq_count {self.custom_seq_count} Batches: {self.__len__()}",
            )

    def sequences_by_user(self, idx: int) -> list[list[int]]:
        return self.sequences[idx]

    def __len__(self) -> int:
        """Amount of available batches to process each sequence at least once."""
        # ! For what a batch means, look at __getitem__().
        # ! There is an externally enforced constrain of "batch size must be lower than the amount of available users"

        if self.usage == Usage.MIN_SEQ_LENGTH:
            # min times amount_of_user_batches:
            return self.min_seq_count * (len(self.users) // self.batch_size)
        if self.usage == Usage.MAX_SEQ_LENGTH:
            # estimated capacity:
            estimated = self.capacity // self.batch_size
            return max(self.max_seq_count, estimated)
        if self.usage == Usage.CUSTOM:
            return self.custom_seq_count * (len(self.users) // self.batch_size)
        raise ValueError()

    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[bool], Tensor]:
        """Against pytorch convention, we directly build a full batch inside __getitem__.
        Use a batch_size of 1 in your pytorch data loader.

        #! The external dataloader object using this dataset will, in each "minibatch", call
        #! __getitem__() only once since batch_size=1 for the dataloader.
        #! Usually the dataloader will call this batch_size number of times and bunch them together
        #! and the above happens for each iteration of the dataloader.
        #! When iterating over the dataloader, a total of dataset __len__() iterations are generated.
        #! Read more at https://stackoverflow.com/a/48611864

        A batch consists of a list of active users,
        their next location sequence with timestamps and coordinates.

        #! There is an externally enforced constrain of "batch size must be lower than the amount of available users"

        y is the target location and y_t, y_s the targets timestamp and coordiantes. Provided for
        possible use.

        reset_h is a flag which indicates when a new user has been replacing a previous user in the
        batch. You should reset this users hidden state to initial value h_0.
        """
        seqs: list[Tensor] = []
        times: list[Tensor] = []
        coords: list[Tensor] = []
        lbls: list[Tensor] = []
        lbl_times: list[Tensor] = []
        lbl_coords: list[Tensor] = []
        reset_h: list[bool] = []
        for i in range(self.batch_size):
            i_user = self.active_users[i]
            j = self.active_user_seq[i]
            max_j = self.sequences_count[i_user]
            if self.usage == Usage.MIN_SEQ_LENGTH:
                max_j = self.min_seq_count
            if self.usage == Usage.CUSTOM:
                max_j = min(
                    max_j, self.custom_seq_count
                )  # use either the users maxima count or limit by custom count
            if j >= max_j:
                # repalce this user in current sequence:
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # ! Remember, there is an externally enforced constrain of "batch size must be lower than the amount of available users"
                # ! so the while loop below will terminate for sure
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))
            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
            self.active_user_seq[i] += 1

        # * Each batch represents a full--length-20--sequence from one user (batch_size different users).
        # * Each time __getitem__() is called we pull the next sequence from the user, or replace that user if they have no more sequences...
        # * ... Think of self.active_users as like a bucket/holding area of users.
        # * stack column wise, so each column is a batch (following RNN convention)
        x = torch.stack(seqs, dim=1)
        # * stack column wise, so each column is a batch (following RNN convention)
        t = torch.stack(times, dim=1)
        # * stack column wise, so each column is a batch (following RNN convention)
        s = torch.stack(coords, dim=1)
        # * stack column wise, so each column is a batch (following RNN convention)
        y = torch.stack(lbls, dim=1)
        # * stack column wise, so each column is a batch (following RNN convention)
        y_t = torch.stack(lbl_times, dim=1)
        # * stack column wise, so each column is a batch (following RNN convention)
        y_s = torch.stack(lbl_coords, dim=1)

        # * x.shape is (sequence_length==20, batch_size)
        # * t.shape is (sequence_length==20, batch_size)
        # * s.shape is (sequence_length==20, batch_size, 2)
        # * y.shape is (sequence_length==20, batch_size)
        # * y_t.shape is (sequence_length==20, batch_size)
        # * y_s.shape is (sequence_length==20, batch_size, 2)
        # * len(reset_h) is batch_size
        # * last return value is a new tensor of the user IDs corresponding to each batch. ~.shape is (batch_size,)
        return x, t, s, y, y_t, y_s, reset_h, torch.tensor(self.active_users)


# * Adapted from our Flashback fork, https://github.com/jackykwe/Flashback_code
class PoiDataloader:
    """Creates datasets from our prepared Gowalla/Foursquare data files.
    The file consist of one check-in per line in the following format (tab separated):

    <user-id> <timestamp> <latitude> <longitude> <location-id>

    Check-ins for the same user have to be on continous lines.
    Ids for users and locations are recreated and continous from 0.
    """

    def __init__(self, loc_count, *, max_users: int = 0, min_checkins: int = 0) -> None:
        """max_users limits the amount of users to load.
        min_checkins discards users with less than this amount of checkins.
        """
        self.max_users: int = max_users
        self.min_checkins: int = min_checkins

        # * maps from client_x.txt's UserId to an internal venue ID used by PoiDataloader and PoiDataset.
        self.user2id: dict[int, int] = {}
        # * maps from client_x.txt's VenueId to an internal venue ID used by PoiDataloader and PoiDataset.
        # ! This definition prevents remapping across different clients in a federated setting.
        # ! The preprocessing already made the locations start indexing from 0
        self.poi2id: dict[int, int] = {i: i for i in range(loc_count)}

        self.users: list[int] = []
        self.times: list[list[float]] = []
        self.coords: list[list[tuple[float, float]]] = []
        self.locs: list[list[int]] = []

    def create_dataset(
        self,
        sequence_length: int,
        batch_size: int,
        split: Split,
        usage: Usage = Usage.MAX_SEQ_LENGTH,
        custom_seq_count: int = 1,
    ):
        return PoiDataset(
            self.users.copy(),
            self.times.copy(),
            self.coords.copy(),
            self.locs.copy(),
            sequence_length,
            batch_size,
            split,
            usage,
            len(self.poi2id),
            custom_seq_count,
        )

    def user_count(self) -> int:
        return len(self.users)

    def locations(self) -> int:
        return len(self.poi2id)

    def read(self, file: Path) -> None:
        if not file.is_file():
            log(
                logging.ERROR,
                f"[Error]: Dataset not available: {file}. Please follow instructions under ./data/README.md",
            )
            sys.exit(1)

        # collect all users with min checkins:
        self.read_users(file)
        # collect checkins for all collected users:
        self.read_pois(file)
        # log(logging.DEBUG, f"self.user2id={self.user2id}")  # * DEBUG SET A
        # log(logging.DEBUG, f"self.poi2id={self.poi2id}")
        assert all(
            [k == v for k, v in self.poi2id.items()]
        ), f"Mapping invalid, results cannot be combined at the federation server: {self.poi2id}"

    def read_users(self, file: Path) -> None:
        f = open(file)
        lines = f.readlines()

        prev_user = int(lines[0].split("\t")[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split("\t")
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                else:
                    raise Exception(
                        "Pre-processing step did not eliminate user with insufficient checkins. Please run `poetry run python -m project.task.flashback.dataset_preparation` then retry."
                    )
                #    log(logging.ERROR, 'discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user
                visit_cnt = 1
                if self.max_users > 0 and len(self.user2id) >= self.max_users:
                    break  # restrict to max users
        # ! Original implementation forgot to consider the last user...
        if visit_cnt >= self.min_checkins:
            self.user2id[prev_user] = len(self.user2id)
        else:
            raise Exception(
                "Pre-processing step did not eliminate user with insufficient checkins. Please run `poetry run python -m project.task.flashback.dataset_preparation` then retry."
            )
        #    log(logging.ERROR, 'discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))

    def read_pois(self, file: Path) -> None:
        f = open(file)
        lines = f.readlines()

        # store location ids
        user_time: list[float] = []
        user_coord: list[tuple[float, float]] = []
        user_loc: list[int] = []

        prev_user = int(lines[0].split("\t")[0])
        prev_user = cast(int, self.user2id.get(prev_user))
        for i, line in enumerate(lines):
            tokens = line.strip().split("\t")
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue  # user is not of interrest
            user = cast(int, self.user2id.get(user))

            time = (
                datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")
                - datetime(1970, 1, 1)
            ).total_seconds()  # unix seconds
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)

            location: int = int(tokens[4])  # location nr
            if self.poi2id.get(location) is None:  # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
            location = cast(int, self.poi2id.get(location))

            if user == prev_user:
                # insert in front!  #* from start to end of list, chronological order (txt file gives descending order)
                user_time.insert(0, time)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:
                self.users.append(prev_user)
                self.times.append(user_time)
                self.coords.append(user_coord)
                self.locs.append(user_loc)

                # resart:
                prev_user = user
                user_time = [time]
                user_coord = [coord]
                user_loc = [location]

        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.times.append(user_time)
        self.coords.append(user_coord)
        self.locs.append(user_loc)


def get_dataloader_generators(
    partition_dir: Path,
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset.

    Parameters
    ----------
    partition_dir : Path
        The path to the partition directory.
        Containing the training data of clients.
        Partitioned by client id.

    Returns
    -------
    Tuple[ClientDataloaderGen, FedDataloaderGen]
        A tuple of functions that return a DataLoader for a client's dataset
        and a DataLoader for the federated dataset.
    """

    def get_client_dataloader(
        cid: CID,
        test: bool,
        _config: dict,  # * this is task.fit_config.dataloader_config or task.eval_config.dataloader_config
        rng_tuple: IsolatedRNG,
    ) -> DataLoader:
        """Return a DataLoader for a client's dataset.

        Parameters
        ----------
        cid : str|int
            The client's ID
        test : bool
            Whether to load the test set or not
        _config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
        DataLoader
            The DataLoader for the client's dataset
        """
        config: ClientDataloaderConfig = ClientDataloaderConfig(**_config)
        del _config

        torch_cpu_generator = rng_tuple[3]

        poi_loader = PoiDataloader(
            config.loc_count, min_checkins=5 * config.sequence_length + 1
        )
        log(
            logging.WARNING,
            f"get_client_dataloader() called, and poi_loader is reading from {partition_dir}...",
        )
        poi_loader.read(partition_dir / f"client_{cid}.txt")
        dataset = poi_loader.create_dataset(
            config.sequence_length,
            config.batch_size,
            Split.TEST if test else Split.TRAIN,
        )
        assert (
            config.batch_size
            <= poi_loader.user_count()  # ! Should be ok, changed < to <=
        ), f"batch size ({config.batch_size}) must be lower than the amount of available users ({poi_loader.user_count()}); cid={cid}, test={test}, _config={config}"
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            generator=torch_cpu_generator,
        )

    # * This should load the global dataset, I think
    def get_federated_dataloader(
        test: bool,
        _config: dict,  # * this is task.fed_test_config.dataloader_config
        rng_tuple: IsolatedRNG,
    ) -> DataLoader:
        """Return a DataLoader for federated train/test sets.

        Parameters
        ----------
        test : bool
            Whether to load the test set or not
        config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
            DataLoader
            The DataLoader for the federated dataset
        """
        config: FedDataloaderConfig = FedDataloaderConfig(
            **_config,
        )
        del _config
        torch_cpu_generator = rng_tuple[3]

        if not test:
            raise NotImplementedError  # Just a guard in case test=False is used

        poi_loader = PoiDataloader(
            config.loc_count, min_checkins=5 * config.sequence_length + 1
        )
        log(
            logging.WARNING,
            f"get_federated_dataloader() called, and poi_loader is reading from {partition_dir.resolve().parent / 'centralised' / 'client_0.txt'}...",
        )
        # Ensure we are reading from the centralised dataset
        poi_loader.read(partition_dir.resolve().parent / "centralised" / "client_0.txt")
        dataset = poi_loader.create_dataset(
            config.sequence_length,
            config.batch_size,
            Split.TEST if test else Split.TRAIN,
        )
        assert (
            config.batch_size < poi_loader.user_count()
        ), f"batch size ({config.batch_size}) must be lower than the amount of available users ({poi_loader.user_count()}); test={test}, _config={config}"
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            generator=torch_cpu_generator,
        )

    return get_client_dataloader, get_federated_dataloader
