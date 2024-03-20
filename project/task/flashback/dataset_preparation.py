"""Functions for Flashback-MCMG download and processing."""

import json
import logging
import shutil
import subprocess
from pathlib import Path

import hydra
import pandas as pd
from flwr.common.logger import log
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

# !This is replaced by `required_minimum_checkins` below
# MIN_CHECKINS = 101  # from Flashback repository's setting.py; reject users with fewer than this number of checkins
# With 101 checkins, the first 80 will go to train (which results in length-79 X-y pairs due to the off-by-one alignment,
#                        i.e. t=0 to t=T-2 used as X, and t=1 to t=T-1 used as the corresponding ys. This shortens the
#                        sequence from length T to length T-1)
#                    and the latter 21 will go to test (which results in length-20 X-y pairs, length-20 is the minimum
#                        for existence of the user in the test set)


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


def _preprocess_data(
    raw_dataset_dir: Path, postprocessed_partitions_root: Path, sequence_length: int
) -> None:
    """
    Preprocesses the downloaded Flashback-MCMG dataset to fit Flashback's expected format.

    Will produce folders
    - postprocessed_partitions_root/<city>/centralised (contains only client_0.txt): centralised
    - postprocessed_partitions_root/<city>/fed_natural (contains client_{0,...,N-1}.txt): natural partition by user
    for each <city>

    Note that only users with at least
        5 * sequence_length + 1
    checkins are retained; the rest are dropped.
    This is a requirement by Flashback. See documentation of the PoiDataset class.
    """
    for city in tqdm(("CAL", "NY", "PHO", "SIN"), desc="Preprocessing"):
        csv_path = raw_dataset_dir / city / f"{city}_checkin.csv"
        # the following is not used for training but potentially useful in a real world use case,
        # to map Flashback-internal location IDs back to FourSquare `VenueId`s
        postprocessed_reverse_mapper_path = (
            postprocessed_partitions_root / city / f"{city}_checkin_reverse_mapper.json"
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
            f"[{city}] After dropping users with less than {required_minimum_checkins} checkins:",
        )
        log(
            logging.INFO,
            f"    {debug_total_checkins_after_drop}/{debug_total_checkins_before_drop} checkins remain ({debug_total_checkins_after_drop * 100 / debug_total_checkins_before_drop:.2f}%)",
        )
        log(
            logging.INFO,
            f"    {debug_total_users_after_drop}/{debug_total_users_before_drop} users remain ({debug_total_users_after_drop * 100 / debug_total_users_before_drop:.2f}%)",
        )

        (postprocessed_partitions_root / city / "centralised").mkdir(
            parents=True, exist_ok=True
        )
        (postprocessed_partitions_root / city / "fed_natural").mkdir(
            parents=True, exist_ok=True
        )

        # Centralised dataset
        df.to_csv(
            postprocessed_partitions_root / city / "centralised" / "client_0.txt",
            sep="\t",
            header=False,
            index=False,
        )

        # Natural partition
        for user_id in df.loc[:, "UserId"].unique():
            df_single_user = df.loc[df.loc[:, "UserId"] == user_id, :]
            df_single_user.to_csv(
                postprocessed_partitions_root
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
    _preprocess_data(
        raw_dataset_dir=Path(cfg.dataset.raw_dataset_dir),
        postprocessed_partitions_root=Path(cfg.dataset.postprocessed_partitions_root),
        sequence_length=cfg.dataset.sequence_length,
    )

    # You should obtain numbers like the following:
    """
    Preprocessing:   0%|                                                                                                | 0/4 [00:00<?, ?it/s]
    [2024-03-19 23:58:16,019][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-19 23:58:16,079][flwr][INFO] - [CAL] After dropping users with less than 101 checkins:
    [2024-03-19 23:58:16,079][flwr][INFO] -     5742/9317 checkins remain (61.63%)
    [2024-03-19 23:58:16,079][flwr][INFO] -     27/130 users remain (20.77%)
    Preprocessing:  25%|██████████████████████                                                                  | 1/4 [00:00<00:00,  4.24it/s]
    [2024-03-19 23:58:20,508][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-19 23:58:48,944][flwr][INFO] - [NY] After dropping users with less than 101 checkins:
    [2024-03-19 23:58:48,944][flwr][INFO] -     225202/430000 checkins remain (52.37%)
    [2024-03-19 23:58:48,944][flwr][INFO] -     1060/8857 users remain (11.97%)
    Preprocessing:  50%|████████████████████████████████████████████                                            | 2/4 [00:35<00:41, 20.60s/it]
    [2024-03-19 23:58:51,413][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-19 23:58:51,748][flwr][INFO] - [PHO] After dropping users with less than 101 checkins:
    [2024-03-19 23:58:51,748][flwr][INFO] -     17723/35337 checkins remain (50.15%)
    [2024-03-19 23:58:51,749][flwr][INFO] -     92/767 users remain (11.99%)
    Preprocessing:  75%|██████████████████████████████████████████████████████████████████                      | 3/4 [00:36<00:11, 11.62s/it]
    [2024-03-19 23:58:54,923][flwr][INFO] - With sequence_length=20, we have required_minimum_checkins=101
    [2024-03-19 23:59:06,297][flwr][INFO] - [SIN] After dropping users with less than 101 checkins:
    [2024-03-19 23:59:06,297][flwr][INFO] -     187093/308136 checkins remain (60.72%)
    [2024-03-19 23:59:06,297][flwr][INFO] -     866/4638 users remain (18.67%)
    Preprocessing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:52<00:00, 13.01s/it]
    """


if __name__ == "__main__":
    download_and_preprocess()
