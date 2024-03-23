"""Functions for Flashback-MCMG download and processing."""

import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flwr.common.logger import log
from numpy.typing import ArrayLike
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


def find_mode(distribution: ArrayLike) -> float:
    """
    Finds mode of the distribution using KDE.
    """
    histplot = sns.histplot(data=distribution, kde=True, element="step")  # corresponds to histplot.get_lines()[0]
    plt.close()  # prevent showing of plot; to see how this function works: comment this line, then call this function

    # Mode finding from KDE courtesy of https://stackoverflow.com/a/72222126
    x, y = histplot.get_lines()[0].get_data()
    mode = x[np.argmax(y)]

    # histplot.axvline(mode, ls="--", color="black")  # to see how this function works: uncomment this line, then call this function
    return mode

def _download_data(
    raw_dataset_dir: Path,
) -> None:
    """Download (if necessary) the Flashback-MCMG dataset."""
    raw_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Assumes this method runs to completion; if this method fails halfway please delete dataset_dir and restart
    # otherwise it may break (handling this properly is work for the future)
    # Check before cloning and unzipping (slow!)
    if not all((
        (raw_dataset_dir / "CAL").exists(),
        (raw_dataset_dir / "NY").exists(),
        (raw_dataset_dir / "PHO").exists(),
        (raw_dataset_dir / "SIN").exists(),
    )):
        # Temporarily clone MCMG repository which contains the datasets
        subprocess.check_call((
            "git",
            "clone",
            "--single-branch",
            "https://github.com/2022MCMG/MCMG.git",
            raw_dataset_dir / "MCMG",
        ))
        for city in ("CAL", "NY", "PHO", "SIN"):
            shutil.move(
                raw_dataset_dir / "MCMG" / "dataset" / city, raw_dataset_dir / city
            )
            if city != "CAL":
                rar_path = raw_dataset_dir / city / f"{city}.rar"
                subprocess.check_call(
                    ("unrar", "e", rar_path, raw_dataset_dir / city)
                )  # extract city/city.rar into city/*.txt
                rar_path.unlink()
        # Delete cloned MCMG repository
        shutil.rmtree(raw_dataset_dir / "MCMG")
    else:
        log(logging.INFO, "Dataset already downloaded.")

def _preprocess_data_all_clients(
    raw_dataset_dir: Path, postprocessed_partitions_root: Path, sequence_length: int
) -> None:
    """
    Preprocesses the downloaded Flashback-MCMG dataset to fit Flashback's expected format.

    Will produce folders
    - postprocessed_partitions_root/all_clients/<city>/centralised (contains only client_0.txt): centralised
    - postprocessed_partitions_root/all_clients/<city>/fed_natural (contains client_{0,...,N-1}.txt): natural partition by user
    for each <city>

    Note that only users with at least
        5 * sequence_length + 1
    checkins are retained; the rest are dropped.
    This is a requirement by Flashback. See documentation of the PoiDataset class.
    """
    for city in tqdm(("CAL", "NY", "PHO", "SIN"), desc="Preprocessing (all_clients)"):
        csv_path = raw_dataset_dir / city / f"{city}_checkin.csv"
        # the following is not used for training but potentially useful in a real world use case,
        # to map Flashback-internal location IDs back to FourSquare `VenueId`s
        postprocessed_reverse_mapper_path = (
            postprocessed_partitions_root / "all_clients" / city / f"{city}_checkin_reverse_mapper.json"
        )

        if postprocessed_reverse_mapper_path.exists():
            log(logging.INFO, f"City {city} already preprocessed.")
            continue

        df = pd.read_csv(csv_path)
        df = df.loc[:, ["UserId", "Local_Time", "Latitude", "Longitude", "VenueId"]]

        # Deal with time
        df.loc[:, "Local_Time"] = pd.to_datetime(
            df.loc[:, "Local_Time"],
            dayfirst=city == "CAL",  # peculiarity of the MCMG dataset
            yearfirst=city != "CAL",  # peculiarity of the MCMG dataset
            utc=True,
        )
        df.loc[:, "Local_Time"] = df.loc[:, "Local_Time"].apply(
            lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        df = df.sort_values(
            by=["UserId", "Local_Time"]
        )  # Flashback expects checkins from the same user to be bunched together

        debug_total_checkins_before_drop = len(df)
        debug_total_users_before_drop = len(df.loc[:, "UserId"].unique())

        # Drop users with insufficient checkins
        required_minimum_checkins = 5 * sequence_length + 1
        log(
            logging.INFO,
            f"With sequence_length={sequence_length}, we have required_minimum_checkins={required_minimum_checkins}",
        )
        num_checkins_per_user = df.loc[:, "UserId"].value_counts()
        df.loc[:, "num_checkins_from_this_user"] = df.loc[:, "UserId"].apply(
            lambda i: num_checkins_per_user[i]
        )
        df = df.loc[
            df.loc[:, "num_checkins_from_this_user"] >= required_minimum_checkins,
            df.columns != "num_checkins_from_this_user",
        ]
        # Mapping string `VenueId`s to integer IDs starting from 0, as expected by Flashback
        venue_ids = list(df.loc[:, "VenueId"].unique())
        df.loc[:, "VenueId"] = df.loc[:, "VenueId"].map(venue_ids.index)
        # Mapping arbitrary integer `UserId`s to integer client IDs starting from 0, as expected by Flower
        user_ids = list(df.loc[:, "UserId"].unique())
        df.loc[:, "UserId"] = df.loc[:, "UserId"].map(user_ids.index)

        debug_total_checkins_after_drop = len(df)
        debug_total_users_after_drop = len(df.loc[:, "UserId"].unique())

        # Print some debugging numbers to see how much of the original MCMG dataset was lost due to this processing
        log(
            logging.INFO,
            f"[{city}: all_clients] After dropping users with less than {required_minimum_checkins} checkins:",
        )
        log(
            logging.INFO,
            f"    {debug_total_checkins_after_drop}/{debug_total_checkins_before_drop} checkins remain ({debug_total_checkins_after_drop * 100 / debug_total_checkins_before_drop:.2f}%)",
        )
        log(
            logging.INFO,
            f"    {debug_total_users_after_drop}/{debug_total_users_before_drop} users remain ({debug_total_users_after_drop * 100 / debug_total_users_before_drop:.2f}%)",
        )

        (postprocessed_partitions_root / "all_clients" / city / "centralised").mkdir(
            parents=True, exist_ok=True
        )
        (postprocessed_partitions_root / "all_clients" / city / "fed_natural").mkdir(
            parents=True, exist_ok=True
        )

        # Centralised dataset
        df.to_csv(
            postprocessed_partitions_root / "all_clients" / city / "centralised" / "client_0.txt",
            sep="\t",
            header=False,
            index=False,
        )

        # Natural partition
        for user_id in df.loc[:, "UserId"].unique():
            df_single_user = df.loc[df.loc[:, "UserId"] == user_id, :]
            df_single_user.to_csv(
                postprocessed_partitions_root
                / "all_clients"
                / city
                / "fed_natural"
                / f"client_{user_id}.txt",
                sep="\t",
                header=False,
                index=False,
            )
            # Datasets for single client training (non-federated setup)
            (postprocessed_partitions_root / "all_clients_nonfederated" / f"{city}-{user_id}" / "centralised").mkdir(
                parents=True, exist_ok=True
            )
            df_single_user.to_csv(
                postprocessed_partitions_root
                / "all_clients_nonfederated"
                / f"{city}-{user_id}"
                / f"centralised"
                / f"client_0.txt",
                sep="\t",
                header=False,
                index=False,
            )

        with postprocessed_reverse_mapper_path.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "venue_id_reverse_mapper": {
                            i: str_venue_id for i, str_venue_id in enumerate(venue_ids)
                        },
                        "user_id_reverse_mapper": {
                            i: int(not_zero_starting_user_id)
                            for i, not_zero_starting_user_id in enumerate(user_ids)
                        },
                    },
                    indent=4,
                )
            )

def _preprocess_data_smaller_quantity_skew(
    raw_dataset_dir: Path, postprocessed_partitions_root: Path, sequence_length: int
) -> None:
    """
    Preprocesses the downloaded Flashback-MCMG dataset to fit Flashback's expected format.
    Only clients within mu +- sigma (in the client dataset size distribution) are retained.

    Will produce folders
    - postprocessed_partitions_root/smaller_quantity_skew/<city>/centralised (contains only client_0.txt): centralised
    - postprocessed_partitions_root/smaller_quantity_skew/<city>/fed_natural (contains client_{0,...,N-1}.txt): natural partition by user
    for each <city>

    Note that only users with at least
        5 * sequence_length + 1
    checkins are retained; the rest are dropped.
    This is a requirement by Flashback. See documentation of the PoiDataset class.
    """
    for city in tqdm(("CAL", "NY", "PHO", "SIN"), desc="Preprocessing (smaller_quantity_skew)"):
        csv_path = raw_dataset_dir / city / f"{city}_checkin.csv"
        # the following is not used for training but potentially useful in a real world use case,
        # to map Flashback-internal location IDs back to FourSquare `VenueId`s
        postprocessed_reverse_mapper_path = (
            postprocessed_partitions_root / "smaller_quantity_skew" / city / f"{city}_checkin_reverse_mapper.json"
        )

        if postprocessed_reverse_mapper_path.exists():
            log(logging.INFO, f"City {city} already preprocessed.")
            continue

        df = pd.read_csv(csv_path)
        df = df.loc[:, ["UserId", "Local_Time", "Latitude", "Longitude", "VenueId"]]

        # Deal with time
        df.loc[:, "Local_Time"] = pd.to_datetime(
            df.loc[:, "Local_Time"],
            dayfirst=city == "CAL",  # peculiarity of the MCMG dataset
            yearfirst=city != "CAL",  # peculiarity of the MCMG dataset
            utc=True,
        )
        df.loc[:, "Local_Time"] = df.loc[:, "Local_Time"].apply(
            lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        df = df.sort_values(
            by=["UserId", "Local_Time"]
        )  # Flashback expects checkins from the same user to be bunched together

        debug_total_checkins_before_drop = len(df)
        debug_total_users_before_drop = len(df.loc[:, "UserId"].unique())

        # Drop users with insufficient checkins
        required_minimum_checkins = 5 * sequence_length + 1
        log(
            logging.INFO,
            f"With sequence_length={sequence_length}, we have required_minimum_checkins={required_minimum_checkins}",
        )
        num_checkins_per_user = df.loc[:, "UserId"].value_counts()
        df.loc[:, "num_checkins_from_this_user"] = df.loc[:, "UserId"].apply(
            lambda i: num_checkins_per_user[i]
        )
        df = df.loc[
            df.loc[:, "num_checkins_from_this_user"] >= required_minimum_checkins,
            :  #* this column is useful for within-barricade operations below, retain it for now
        ]

        # * Extra processing compared to _preprocess_data_all_clients() happens within
        # * these barricades
        # * //////////////////////////////////////////////////////////////////////// * #

        # Checkins distribution
        #! Remember to re-evaluate after dropping clients above!
        num_checkins_per_user = df.loc[:, "UserId"].value_counts()
        mu = np.mean(num_checkins_per_user)
        sigma = np.std(num_checkins_per_user)  # standard deviation

        # Keep only clients with number of checkins inside [mu - sigma, mu + sigma] (both inclusive; liberally allow more otherwise small datasets will have problem...)
        accept = (df.loc[:, "num_checkins_from_this_user"] >= mu - sigma) & (df.loc[:, "num_checkins_from_this_user"] <= mu + sigma)
        df = df.loc[
            accept,
            df.columns != "num_checkins_from_this_user",
        ]

        # * //////////////////////////////////////////////////////////////////////// * #

        # Mapping string `VenueId`s to integer IDs starting from 0, as expected by Flashback
        venue_ids = list(df.loc[:, "VenueId"].unique())
        df.loc[:, "VenueId"] = df.loc[:, "VenueId"].map(venue_ids.index)
        # Mapping arbitrary integer `UserId`s to integer client IDs starting from 0, as expected by Flower
        user_ids = list(df.loc[:, "UserId"].unique())
        df.loc[:, "UserId"] = df.loc[:, "UserId"].map(user_ids.index)

        debug_total_checkins_after_drop = len(df)
        debug_total_users_after_drop = len(df.loc[:, "UserId"].unique())

        # Print some debugging numbers to see how much of the original MCMG dataset was lost due to this processing
        log(
            logging.INFO,
            f"[{city}: smaller_quantity_skew] After dropping users with less than {required_minimum_checkins} checkins, then those not near the resultant mean:",
        )
        log(
            logging.INFO,
            f"    {debug_total_checkins_after_drop}/{debug_total_checkins_before_drop} checkins remain ({debug_total_checkins_after_drop * 100 / debug_total_checkins_before_drop:.2f}%)",
        )
        log(
            logging.INFO,
            f"    {debug_total_users_after_drop}/{debug_total_users_before_drop} users remain ({debug_total_users_after_drop * 100 / debug_total_users_before_drop:.2f}%)",
        )

        (postprocessed_partitions_root / "smaller_quantity_skew" / city / "centralised").mkdir(
            parents=True, exist_ok=True
        )
        (postprocessed_partitions_root / "smaller_quantity_skew" / city / "fed_natural").mkdir(
            parents=True, exist_ok=True
        )

        # Centralised dataset
        df.to_csv(
            postprocessed_partitions_root / "smaller_quantity_skew" / city / "centralised" / "client_0.txt",
            sep="\t",
            header=False,
            index=False,
        )

        # Natural partition
        for user_id in df.loc[:, "UserId"].unique():
            df_single_user = df.loc[df.loc[:, "UserId"] == user_id, :]
            df_single_user.to_csv(
                postprocessed_partitions_root
                / "smaller_quantity_skew"
                / city
                / "fed_natural"
                / f"client_{user_id}.txt",
                sep="\t",
                header=False,
                index=False,
            )

        with postprocessed_reverse_mapper_path.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "venue_id_reverse_mapper": {
                            i: str_venue_id for i, str_venue_id in enumerate(venue_ids)
                        },
                        "user_id_reverse_mapper": {
                            i: int(not_zero_starting_user_id)
                            for i, not_zero_starting_user_id in enumerate(user_ids)
                        },
                    },
                    indent=4,
                )
            )

def _preprocess_data_homogeneous_approximation(
    raw_dataset_dir: Path, postprocessed_partitions_root: Path, sequence_length: int
) -> None:
    """
    Preprocesses the downloaded Flashback-MCMG dataset to fit Flashback's expected format.
    Only clients within mu +- sigma (in the client dataset size distribution) are retained.

    Will produce folders
    - postprocessed_partitions_root/homogeneous_approximation/<city>/centralised (contains only client_0.txt): centralised
    - postprocessed_partitions_root/homogeneous_approximation/<city>/fed_natural (contains client_{0,...,N-1}.txt): natural partition by user
    for each <city>

    Note that only users with at least
        5 * sequence_length + 1
    checkins are retained; the rest are dropped.
    This is a requirement by Flashback. See documentation of the PoiDataset class.
    """
    for city in tqdm(("CAL", "NY", "PHO", "SIN"), desc="Preprocessing (homogeneous_approximation)"):
        csv_path = raw_dataset_dir / city / f"{city}_checkin.csv"
        # the following is not used for training but potentially useful in a real world use case,
        # to map Flashback-internal location IDs back to FourSquare `VenueId`s
        postprocessed_reverse_mapper_path = (
            postprocessed_partitions_root / "homogeneous_approximation" / city / f"{city}_checkin_reverse_mapper.json"
        )

        if postprocessed_reverse_mapper_path.exists():
            log(logging.INFO, f"City {city} already preprocessed.")
            continue

        df = pd.read_csv(csv_path)
        df = df.loc[:, ["UserId", "Local_Time", "Latitude", "Longitude", "VenueId"]]

        # Deal with time
        df.loc[:, "Local_Time"] = pd.to_datetime(
            df.loc[:, "Local_Time"],
            dayfirst=city == "CAL",  # peculiarity of the MCMG dataset
            yearfirst=city != "CAL",  # peculiarity of the MCMG dataset
            utc=True,
        )
        df.loc[:, "Local_Time"] = df.loc[:, "Local_Time"].apply(
            lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        df = df.sort_values(
            by=["UserId", "Local_Time"]
        )  # Flashback expects checkins from the same user to be bunched together

        debug_total_checkins_before_drop = len(df)
        debug_total_users_before_drop = len(df.loc[:, "UserId"].unique())

        # Drop users with insufficient checkins
        required_minimum_checkins = 5 * sequence_length + 1
        log(
            logging.INFO,
            f"With sequence_length={sequence_length}, we have required_minimum_checkins={required_minimum_checkins}",
        )
        num_checkins_per_user = df.loc[:, "UserId"].value_counts()
        df.loc[:, "num_checkins_from_this_user"] = df.loc[:, "UserId"].apply(
            lambda i: num_checkins_per_user[i]
        )
        df = df.loc[
            df.loc[:, "num_checkins_from_this_user"] >= required_minimum_checkins,
            df.columns != "num_checkins_from_this_user",
        ]

        # * Extra processing compared to _preprocess_data_all_clients() happens within
        # * these barricades
        # * //////////////////////////////////////////////////////////////////////// * #

        # Checkins distribution
        #! Remember to re-evaluate after dropping clients above!
        num_checkins_per_user = df.loc[:, "UserId"].value_counts()
        mode = find_mode(num_checkins_per_user)
        fifteen_percent_of_users = math.ceil(0.15 * len(num_checkins_per_user))
        users_to_keep = (num_checkins_per_user - mode).abs().sort_values(kind="stable")[:fifteen_percent_of_users].index  # .index valid because num_checkins_per_user is a pd.Series

        # Keep only 15% of all clients: those that are closest to the mode
        df = df.loc[
            df.loc[:, "UserId"].isin(users_to_keep),
            :
        ]

        # * //////////////////////////////////////////////////////////////////////// * #

        # Mapping string `VenueId`s to integer IDs starting from 0, as expected by Flashback
        venue_ids = list(df.loc[:, "VenueId"].unique())
        df.loc[:, "VenueId"] = df.loc[:, "VenueId"].map(venue_ids.index)
        # Mapping arbitrary integer `UserId`s to integer client IDs starting from 0, as expected by Flower
        user_ids = list(df.loc[:, "UserId"].unique())
        df.loc[:, "UserId"] = df.loc[:, "UserId"].map(user_ids.index)

        debug_total_checkins_after_drop = len(df)
        debug_total_users_after_drop = len(df.loc[:, "UserId"].unique())

        # Print some debugging numbers to see how much of the original MCMG dataset was lost due to this processing
        log(
            logging.INFO,
            f"[{city}: homogeneous_approximation] After dropping users with less than {required_minimum_checkins} checkins, then those not very near the resultant mode:",
        )
        log(
            logging.INFO,
            f"    {debug_total_checkins_after_drop}/{debug_total_checkins_before_drop} checkins remain ({debug_total_checkins_after_drop * 100 / debug_total_checkins_before_drop:.2f}%)",
        )
        log(
            logging.INFO,
            f"    {debug_total_users_after_drop}/{debug_total_users_before_drop} users remain ({debug_total_users_after_drop * 100 / debug_total_users_before_drop:.2f}%)",
        )

        (postprocessed_partitions_root / "homogeneous_approximation" / city / "centralised").mkdir(
            parents=True, exist_ok=True
        )
        (postprocessed_partitions_root / "homogeneous_approximation" / city / "fed_natural").mkdir(
            parents=True, exist_ok=True
        )

        # Centralised dataset
        df.to_csv(
            postprocessed_partitions_root / "homogeneous_approximation" / city / "centralised" / "client_0.txt",
            sep="\t",
            header=False,
            index=False,
        )

        # Natural partition
        for user_id in df.loc[:, "UserId"].unique():
            df_single_user = df.loc[df.loc[:, "UserId"] == user_id, :]
            df_single_user.to_csv(
                postprocessed_partitions_root
                / "homogeneous_approximation"
                / city
                / "fed_natural"
                / f"client_{user_id}.txt",
                sep="\t",
                header=False,
                index=False,
            )

        with postprocessed_reverse_mapper_path.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "venue_id_reverse_mapper": {
                            i: str_venue_id for i, str_venue_id in enumerate(venue_ids)
                        },
                        "user_id_reverse_mapper": {
                            i: int(not_zero_starting_user_id)
                            for i, not_zero_starting_user_id in enumerate(user_ids)
                        },
                    },
                    indent=4,
                )
            )

def _print_preprocessing_statistics(
    raw_dataset_dir: Path, postprocessed_partitions_root: Path, sequence_length: int
) -> None:
    """
    Reprints stuff that would have been printed in the above functions (useful if
    rerunning task.dataset_preparation)
    """
    for heterogeneity in ("all_clients", "smaller_quantity_skew", "homogeneous_approximation"):
        for city in ("CAL", "NY", "PHO", "SIN"):
            csv_path = raw_dataset_dir / city / f"{city}_checkin.csv"
            df_raw = pd.read_csv(csv_path)
            df_raw = df_raw.loc[:, ["UserId", "Local_Time", "Latitude", "Longitude", "VenueId"]]
            debug_total_checkins_before_drop = len(df_raw)
            debug_total_users_before_drop = len(df_raw.loc[:, "UserId"].unique())

            csv_path = postprocessed_partitions_root / heterogeneity / city / "centralised" / "client_0.txt"
            df_postprocessed = pd.read_csv(csv_path, sep="\t", header=None)
            debug_total_checkins_after_drop = len(df_postprocessed)
            debug_total_users_after_drop = len(df_postprocessed.loc[:, 0].unique())

            required_minimum_checkins = 5 * sequence_length + 1

            if heterogeneity == "all_clients":
                descriptor = f"After dropping users with less than {required_minimum_checkins} checkins"
            elif heterogeneity == "smaller_quantity_skew":
                descriptor = f"After dropping users with less than {required_minimum_checkins} checkins, then those not near the resultant mean"
            elif heterogeneity == "homogeneous_approximation":
                descriptor = f"After dropping users with less than {required_minimum_checkins} checkins, then those not very near the resultant mode"
            else:
                raise NotImplementedError
            log(
                logging.INFO,
                f"[{city}: {heterogeneity}] {descriptor}:",
            )
            log(
                logging.INFO,
                f"    {debug_total_checkins_after_drop}/{debug_total_checkins_before_drop} checkins remain ({debug_total_checkins_after_drop * 100 / debug_total_checkins_before_drop:.2f}%)",
            )
            log(
                logging.INFO,
                f"    {debug_total_users_after_drop}/{debug_total_users_before_drop} users remain ({debug_total_users_after_drop * 100 / debug_total_users_before_drop:.2f}%)",
            )

def _print_statistics(postprocessed_partitions_root: Path) -> None:
    """
    Prints number of users and locations in each dataset.
    """
    for heterogeneity in ("all_clients", "smaller_quantity_skew", "homogeneous_approximation"):
        for city in ("CAL", "NY", "PHO", "SIN"):
            csv_path = postprocessed_partitions_root / heterogeneity / city / "centralised" / "client_0.txt"
            df = pd.read_csv(csv_path, sep="\t", header=None)
            user_count = len(df.loc[:, 0].unique())
            loc_count = len(df.loc[:, 4].unique())
            log(
                logging.INFO, f"[{heterogeneity}] city={city}: user_count={user_count}, loc_count={loc_count}"
            )


@hydra.main(
    config_path="../../conf",
    config_name="flashback",
    version_base=None,
)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and preprocess the dataset.

    Please include here all the logic
    Please use the Hydra config style as much as possible specially
    for parts that can be customized (e.g. how data is partitioned)

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    # Download the dataset
    _download_data(
        raw_dataset_dir=Path(cfg.dataset.raw_dataset_dir),
    )

    # Preprocess the datasets to fit Flashback's expected format
    for fn in (
        _preprocess_data_all_clients,
        _preprocess_data_smaller_quantity_skew,
        _preprocess_data_homogeneous_approximation,
        _print_preprocessing_statistics
    ):
        fn(
            raw_dataset_dir=Path(cfg.dataset.raw_dataset_dir),
            postprocessed_partitions_root=Path(cfg.dataset.postprocessed_partitions_root),
            sequence_length=cfg.dataset.sequence_length,
        )

    _print_statistics(
        postprocessed_partitions_root=Path(cfg.dataset.postprocessed_partitions_root)
    )

    # You should obtain numbers like the following:
    """
    Preprocessing (all_clients):   0%|                                                                                  | 0/4 [00:00<?, ?it/s]
    [2024-03-23 05:23:14,825][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:23:14,887][flwr][INFO] - [CAL: all_clients] After dropping users with less than 101 checkins:
    [2024-03-23 05:23:14,888][flwr][INFO] -     5742/9317 checkins remain (61.63%)
    [2024-03-23 05:23:14,888][flwr][INFO] -     27/130 users remain (20.77%)
    Preprocessing (all_clients):  25%|██████████████████▌                                                       | 1/4 [00:00<00:00,  3.81it/s]
    [2024-03-23 05:23:20,066][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:23:54,902][flwr][INFO] - [NY: all_clients] After dropping users with less than 101 checkins:
    [2024-03-23 05:23:54,902][flwr][INFO] -     225202/430000 checkins remain (52.37%)
    [2024-03-23 05:23:54,902][flwr][INFO] -     1060/8857 users remain (11.97%)
    Preprocessing (all_clients):  50%|█████████████████████████████████████                                     | 2/4 [00:42<00:49, 24.97s/it]
    [2024-03-23 05:23:57,657][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:23:58,028][flwr][INFO] - [PHO: all_clients] After dropping users with less than 101 checkins:
    [2024-03-23 05:23:58,028][flwr][INFO] -     17723/35337 checkins remain (50.15%)
    [2024-03-23 05:23:58,028][flwr][INFO] -     92/767 users remain (11.99%)
    Preprocessing (all_clients):  75%|███████████████████████████████████████████████████████▌                  | 3/4 [00:43<00:14, 14.02s/it]
    [2024-03-23 05:24:01,570][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:24:13,558][flwr][INFO] - [SIN: all_clients] After dropping users with less than 101 checkins:
    [2024-03-23 05:24:13,558][flwr][INFO] -     187093/308136 checkins remain (60.72%)
    [2024-03-23 05:24:13,558][flwr][INFO] -     866/4638 users remain (18.67%)
    Preprocessing (all_clients): 100%|██████████████████████████████████████████████████████████████████████████| 4/4 [01:00<00:00, 15.17s/it]
    Preprocessing (smaller_quantity_skew):   0%|                                                                        | 0/4 [00:00<?, ?it/s]
    [2024-03-23 05:39:28,708][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:39:28,768][flwr][INFO] - [CAL: smaller_quantity_skew] After dropping users with less than 101 checkins, then those not near the resultant mean:
    [2024-03-23 05:39:28,768][flwr][INFO] -     4713/9317 checkins remain (50.58%)
    [2024-03-23 05:39:28,768][flwr][INFO] -     25/130 users remain (19.23%)
    Preprocessing (smaller_quantity_skew):  25%|████████████████                                                | 1/4 [00:00<00:00,  4.04it/s]
    [2024-03-23 05:39:33,693][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:40:12,433][flwr][INFO] - [NY: smaller_quantity_skew] After dropping users with less than 101 checkins, then those not near the resultant mean:
    [2024-03-23 05:40:12,433][flwr][INFO] -     170852/430000 checkins remain (39.73%)
    [2024-03-23 05:40:12,433][flwr][INFO] -     992/8857 users remain (11.20%)
    Preprocessing (smaller_quantity_skew):  50%|████████████████████████████████                                | 2/4 [00:45<00:53, 26.80s/it]
    [2024-03-23 05:40:14,713][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:40:15,066][flwr][INFO] - [PHO: smaller_quantity_skew] After dropping users with less than 101 checkins, then those not near the resultant mean:
    [2024-03-23 05:40:15,066][flwr][INFO] -     14010/35337 checkins remain (39.65%)
    [2024-03-23 05:40:15,066][flwr][INFO] -     86/767 users remain (11.21%)
    Preprocessing (smaller_quantity_skew):  75%|████████████████████████████████████████████████                | 3/4 [00:46<00:15, 15.02s/it]
    [2024-03-23 05:40:19,828][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:40:45,932][flwr][INFO] - [SIN: smaller_quantity_skew] After dropping users with less than 101 checkins, then those not near the resultant mean:
    [2024-03-23 05:40:45,932][flwr][INFO] -     137346/308136 checkins remain (44.57%)
    [2024-03-23 05:40:45,933][flwr][INFO] -     797/4638 users remain (17.18%)
    Preprocessing (smaller_quantity_skew): 100%|████████████████████████████████████████████████████████████████| 4/4 [01:19<00:00, 19.96s/it]
    Preprocessing (homogeneous_approximation):   0%|                                                                    | 0/4 [00:00<?, ?it/s]
    [2024-03-23 05:40:48,590][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:40:49,256][flwr][INFO] - [CAL: homogeneous_approximation] After dropping users with less than 101 checkins, then those not very near the resultant mode:
    [2024-03-23 05:40:49,256][flwr][INFO] -     822/9317 checkins remain (8.82%)
    [2024-03-23 05:40:49,256][flwr][INFO] -     5/130 users remain (3.85%)
    Preprocessing (homogeneous_approximation):  25%|███████████████                                             | 1/4 [00:00<00:02,  1.19it/s]
    [2024-03-23 05:40:57,301][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:41:00,798][flwr][INFO] - [NY: homogeneous_approximation] After dropping users with less than 101 checkins, then those not very near the resultant mode:
    [2024-03-23 05:41:00,798][flwr][INFO] -     22131/430000 checkins remain (5.15%)
    [2024-03-23 05:41:00,798][flwr][INFO] -     159/8857 users remain (1.80%)
    Preprocessing (homogeneous_approximation):  50%|██████████████████████████████                              | 2/4 [00:12<00:14,  7.29s/it]
    [2024-03-23 05:41:01,512][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:41:01,818][flwr][INFO] - [PHO: homogeneous_approximation] After dropping users with less than 101 checkins, then those not very near the resultant mode:
    [2024-03-23 05:41:01,818][flwr][INFO] -     1968/35337 checkins remain (5.57%)
    [2024-03-23 05:41:01,818][flwr][INFO] -     14/767 users remain (1.83%)
    Preprocessing (homogeneous_approximation):  75%|█████████████████████████████████████████████               | 3/4 [00:13<00:04,  4.32s/it]
    [2024-03-23 05:41:05,937][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-23 05:41:08,328][flwr][INFO] - [SIN: homogeneous_approximation] After dropping users with less than 101 checkins, then those not very near the resultant mode:
    [2024-03-23 05:41:08,329][flwr][INFO] -     18517/308136 checkins remain (6.01%)
    [2024-03-23 05:41:08,329][flwr][INFO] -     130/4638 users remain (2.80%)
    Preprocessing (homogeneous_approximation): 100%|████████████████████████████████████████████████████████████| 4/4 [00:20<00:00,  5.15s/it]
    [2024-03-23 05:43:13,105][flwr][INFO] - [all_clients] city=CAL: user_count=27, loc_count=472
    [2024-03-23 05:43:13,300][flwr][INFO] - [all_clients] city=NY: user_count=1060, loc_count=16015
    [2024-03-23 05:43:13,339][flwr][INFO] - [all_clients] city=PHO: user_count=92, loc_count=1479
    [2024-03-23 05:43:13,500][flwr][INFO] - [all_clients] city=SIN: user_count=866, loc_count=8694
    [2024-03-23 05:43:13,525][flwr][INFO] - [smaller_quantity_skew] city=CAL: user_count=25, loc_count=444
    [2024-03-23 05:43:13,896][flwr][INFO] - [smaller_quantity_skew] city=NY: user_count=992, loc_count=14925
    [2024-03-23 05:43:13,948][flwr][INFO] - [smaller_quantity_skew] city=PHO: user_count=86, loc_count=1360
    [2024-03-23 05:43:14,095][flwr][INFO] - [smaller_quantity_skew] city=SIN: user_count=797, loc_count=7930
    [2024-03-23 05:43:14,101][flwr][INFO] - [homogeneous_approximation] city=CAL: user_count=5, loc_count=138
    [2024-03-23 05:43:14,124][flwr][INFO] - [homogeneous_approximation] city=NY: user_count=159, loc_count=5995
    [2024-03-23 05:43:14,129][flwr][INFO] - [homogeneous_approximation] city=PHO: user_count=14, loc_count=406
    [2024-03-23 05:43:14,147][flwr][INFO] - [homogeneous_approximation] city=SIN: user_count=130, loc_count=3215
    """


if __name__ == "__main__":
    download_and_preprocess()
