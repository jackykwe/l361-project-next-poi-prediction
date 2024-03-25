"""MCMG dataset utilities for federated learning."""
import pickle
import numpy as np
from itertools import compress
from scipy import stats as st

from project.task.default.dataset import (
    ClientDataloaderConfig as DefaultClientDataloaderConfig,
)
from project.task.default.dataset import (
    FedDataloaderConfig as DefaultFedDataloaderConfig,
)
from project.task.mcmg.mcmg_utils import slice_data, increase_data, flatten_data, get_In_Cross_region_seq, \
    get_adj_matrix_InDegree, Data, Data_GroupLabel
from project.types.common import (
    CID,
    ClientDataloaderGen,
    FedDataloaderGen,
    IsolatedRNG,
)

# Use defaults for this very simple dataset
# Requires only batch size
ClientDataloaderConfig = DefaultClientDataloaderConfig
FedDataloaderConfig = DefaultFedDataloaderConfig


def get_dataloader_generators(
        data_dir: str,
        data_name: str,
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset.

    Parameters
    ----------
    data : str
        The MCMG dataset to use.

    Returns
    -------
    Tuple[ClientDataloaderGen, FedDataloaderGen]
        A tuple of functions that return a DataLoader for a client's dataset
        and a DataLoader for the federated dataset.
    """

    def get_client_dataloader(
            cid: CID, test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> object:
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
        object
            The DataLoader for the client's dataset
        """

        cid = int(cid)

        with open(f'{data_dir}/{data_name}/train.pkl', "rb") as f:
            full_train = pickle.load(f)

        with open(f'{data_dir}/{data_name}/test.pkl', "rb") as f:
            full_test = pickle.load(f)

        if _config['filter']:
            full_train, full_test = filter_data_heterogeneous(full_train, full_test)

        train_data = increase_data(flatten_data(full_train))
        POI_adj_matrix = get_adj_matrix_InDegree(train_data[0], _config['POI_n_node'])

        client_idxs = [cid]

        if not test:
            client_train = slice_data(full_train, client_idxs)
            train_data = increase_data(flatten_data(client_train))

            train_regions = get_In_Cross_region_seq(client_train[4])
            group_label_train = flatten_data((train_regions,))[0]

            POI_train_data = Data(train_data[0:2], shuffle=False)
            cate_train_data = Data(train_data[2:4], shuffle=False)
            regi_train_data = Data(train_data[4:6], shuffle=False)
            time_train_data = Data(train_data[6:8], shuffle=False)
            POI_dist_train_data = Data(train_data[8:10], shuffle=False)
            regi_dist_train_data = Data(train_data[10:12], shuffle=False)
            group_label_train = Data_GroupLabel(group_label_train, shuffle=False)

            # We return a custom data object for compatibility with the original MCMG implementation.
            return McmgTrainDataLoader(POI_adj_matrix, POI_train_data, cate_train_data, regi_train_data,
                                       time_train_data,
                                       POI_dist_train_data, regi_dist_train_data, group_label_train)

        else:
            client_test = slice_data(full_test, client_idxs)
            test_data = increase_data(flatten_data(client_test))

            test_regions = get_In_Cross_region_seq(client_test[4])
            group_label_test = flatten_data((test_regions,))[0]

            group_label_test = Data_GroupLabel(group_label_test, shuffle=False)
            POI_test_data = Data(test_data[0:2], shuffle=False)
            cate_test_data = Data(test_data[2:4], shuffle=False)
            regi_test_data = Data(test_data[4:6], shuffle=False)
            time_test_data = Data(test_data[6:8], shuffle=False)
            POI_dist_test_data = Data(test_data[8:10], shuffle=False)
            regi_dist_test_data = Data(test_data[10:12], shuffle=False)

            # We return a custom data object for compatibility with the original MCMG implementation.
            return McmgTestDataLoader(POI_adj_matrix, POI_test_data, cate_test_data, regi_test_data, time_test_data,
                                      POI_dist_test_data,
                                      regi_dist_test_data, group_label_test)

    def get_federated_dataloader(
            test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> object:
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
            object
            The DataLoader for the federated dataset
        """
        with open(f'{data_dir}/{data_name}/train.pkl', "rb") as f:
            full_train = pickle.load(f)

        with open(f'{data_dir}/{data_name}/test.pkl', "rb") as f:
            full_test = pickle.load(f)

        train_data = increase_data(flatten_data(full_train))
        POI_adj_matrix = get_adj_matrix_InDegree(train_data[0], _config['POI_n_node'])

        if not test:
            train_data = increase_data(flatten_data(full_train))

            train_regions = get_In_Cross_region_seq(full_train[4])
            group_label_train = flatten_data((train_regions,))[0]

            POI_train_data = Data(train_data[0:2], shuffle=False)
            cate_train_data = Data(train_data[2:4], shuffle=False)
            regi_train_data = Data(train_data[4:6], shuffle=False)
            time_train_data = Data(train_data[6:8], shuffle=False)
            POI_dist_train_data = Data(train_data[8:10], shuffle=False)
            regi_dist_train_data = Data(train_data[10:12], shuffle=False)
            group_label_train = Data_GroupLabel(group_label_train, shuffle=False)

            # We return a custom data object for compatibility with the original MCMG implementation.
            return McmgTrainDataLoader(POI_adj_matrix, POI_train_data, cate_train_data, regi_train_data,
                                       time_train_data,
                                       POI_dist_train_data, regi_dist_train_data, group_label_train)

        else:
            test_data = increase_data(flatten_data(full_test))

            test_regions = get_In_Cross_region_seq(full_test[4])
            group_label_test = flatten_data((test_regions,))[0]

            group_label_test = Data_GroupLabel(group_label_test, shuffle=False)
            POI_test_data = Data(test_data[0:2], shuffle=False)
            cate_test_data = Data(test_data[2:4], shuffle=False)
            regi_test_data = Data(test_data[4:6], shuffle=False)
            time_test_data = Data(test_data[6:8], shuffle=False)
            POI_dist_test_data = Data(test_data[8:10], shuffle=False)
            regi_dist_test_data = Data(test_data[10:12], shuffle=False)

            # We return a custom data object for compatibility with the original MCMG implementation.
            return McmgTestDataLoader(POI_adj_matrix, POI_test_data, cate_test_data, regi_test_data, time_test_data,
                                      POI_dist_test_data,
                                      regi_dist_test_data, group_label_test)

    return get_client_dataloader, get_federated_dataloader


class McmgTrainDataLoader:

    def __init__(self, POI_adj_matrix, POI_train_data, cate_train_data, regi_train_data, time_train_data,
                 POI_dist_train_data, regi_dist_train_data, group_label_train):
        super().__init__()

        self.POI_adj_matrix = POI_adj_matrix

        self.POI_train_data = POI_train_data
        self.cate_train_data = cate_train_data
        self.regi_train_data = regi_train_data
        self.time_train_data = time_train_data
        self.POI_dist_train_data = POI_dist_train_data
        self.regi_dist_train_data = regi_dist_train_data
        self.group_label_train = group_label_train


class McmgTestDataLoader:

    def __init__(self, POI_adj_matrix, POI_test_data,
                 cate_test_data, regi_test_data, time_test_data, POI_dist_test_data, regi_dist_test_data,
                 group_label_test):
        super().__init__()

        self.POI_adj_matrix = POI_adj_matrix

        self.group_label_test = group_label_test
        self.POI_test_data = POI_test_data
        self.cate_test_data = cate_test_data
        self.regi_test_data = regi_test_data
        self.time_test_data = time_test_data
        self.POI_dist_test_data = POI_dist_test_data
        self.regi_dist_test_data = regi_dist_test_data


def filter_data_heterogeneous(full_train, full_test):
    client_lens = [len(i) for i in full_train[0]]
    mean_client_len = np.mean(client_lens)
    std_client_len = np.std(client_lens)

    filtered_train = list()
    filtered_test = list()

    mask1 = client_lens >= mean_client_len + std_client_len
    mask2 = client_lens <= mean_client_len - std_client_len
    mask = mask1 | mask2

    for i in range(0, len(full_train)):
        filtered_train.append(list(compress(full_train[i], mask)))
        filtered_test.append(list(compress(full_test[i], mask)))

    return filtered_train, filtered_test


def filter_data_homogeneous(full_train, full_test):
    client_lens = [len(i) for i in full_train[0]]
    mode = st.mode(client_lens).mode

    filtered_train = list()
    filtered_test = list()

    mask = client_lens == mode

    for i in range(0, len(full_train)):
        filtered_train.append(list(compress(full_train[i], mask)))
        filtered_test.append(list(compress(full_test[i], mask)))

    return filtered_train, filtered_test
