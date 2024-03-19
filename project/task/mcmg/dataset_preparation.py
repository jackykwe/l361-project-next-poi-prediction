"""Functions for MNIST download and processing."""

import logging
from pathlib import Path

import hydra
from flwr.common.logger import log
from omegaconf import DictConfig, OmegaConf
import shutil
import patoolib
import requests
from mcmg_utils import *
import pickle


def _download_data(mcmg_dir: Path):
    """Download (if necessary) the MCMG datasets."""
    if not os.path.isdir(str(mcmg_dir)):
        print("Downloading data")
        os.makedirs(str(mcmg_dir))
        data = requests.get(
            "https://github.com/2022MCMG/MCMG/archive/refs/heads/main.zip"
        )
        file_name = f"{mcmg_dir!s}/mcmg.zip"

        with open(file_name, "wb") as file:
            file.write(data.content)
        print("Data downloaded")

        print("Extract data")
        shutil.unpack_archive(file_name, str(mcmg_dir))

        shutil.move(
            os.path.join(mcmg_dir, "MCMG-main/dataset/CAL"),
            os.path.join(mcmg_dir, "CAL"),
        )
        patoolib.extract_archive(
            f"{mcmg_dir!s}/MCMG-main/dataset/NY/NY.rar", outdir=str(mcmg_dir)
        )
        patoolib.extract_archive(
            f"{mcmg_dir!s}/MCMG-main/dataset/PHO/PHO.rar", outdir=str(mcmg_dir)
        )
        patoolib.extract_archive(
            f"{mcmg_dir!s}/MCMG-main/dataset/SIN/SIN.rar", outdir=str(mcmg_dir)
        )

        shutil.rmtree(os.path.join(mcmg_dir, "MCMG-main"))
        print("Data extracted")


def _preprocess(cfg):
    # Preprocess the datasets

    cal_dir = cfg.dataset.cal_dir
    if not os.path.isdir(cal_dir):
        os.makedirs(cal_dir)

        df = pd.read_csv(f"{cfg.dataset.mcmg_dir}/CAL/CAL_checkin.csv")
        day_first = True
        train, val, _, test = get_tr_va_te_data(df, day_first)

        with open(f"{cal_dir!s}/train.pkl", "wb") as f:
            pickle.dump(train, f)

        with open(f"{cal_dir!s}/val.pkl", "wb") as f:
            pickle.dump(val, f)

        with open(f"{cal_dir!s}/test.pkl", "wb") as f:
            pickle.dump(test, f)

    ny_dir = cfg.dataset.ny_dir
    if not os.path.isdir(ny_dir):
        os.makedirs(ny_dir)

        df = pd.read_csv(f"{cfg.dataset.mcmg_dir}/NY/NY_checkin.csv")
        day_first = False
        train, val, _, test = get_tr_va_te_data(df, day_first)

        with open(f"{ny_dir!s}/train.pkl", "wb") as f:
            pickle.dump(train, f)

        with open(f"{ny_dir!s}/val.pkl", "wb") as f:
            pickle.dump(val, f)

        with open(f"{ny_dir!s}/test.pkl", "wb") as f:
            pickle.dump(test, f)

    pho_dir = cfg.dataset.pho_dir
    if not os.path.isdir(pho_dir):
        os.makedirs(pho_dir)

        df = pd.read_csv(f"{cfg.dataset.mcmg_dir}/PHO/PHO_checkin.csv")
        day_first = False
        train, val, _, test = get_tr_va_te_data(df, day_first)

        with open(f"{pho_dir!s}/train.pkl", "wb") as f:
            pickle.dump(train, f)

        with open(f"{pho_dir!s}/val.pkl", "wb") as f:
            pickle.dump(val, f)

        with open(f"{pho_dir!s}/test.pkl", "wb") as f:
            pickle.dump(test, f)

    sin_dir = cfg.dataset.sin_dir
    if not os.path.isdir(sin_dir):
        os.makedirs(sin_dir)

        df = pd.read_csv(f"{cfg.dataset.mcmg_dir}/SIN/SIN_checkin.csv")
        day_first = False
        train, val, _, test = get_tr_va_te_data(df, day_first)

        with open(f"{sin_dir!s}/train.pkl", "wb") as f:
            pickle.dump(train, f)

        with open(f"{sin_dir!s}/val.pkl", "wb") as f:
            pickle.dump(val, f)

        with open(f"{sin_dir!s}/test.pkl", "wb") as f:
            pickle.dump(test, f)


@hydra.main(
    config_path="../../conf",
    config_name="mcmg",
    version_base=None,
)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and preprocess the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    # Download the datasets
    _download_data(Path(cfg.dataset.mcmg_dir))

    # Preprocess the datasets
    _preprocess(cfg)


if __name__ == "__main__":
    download_and_preprocess()
