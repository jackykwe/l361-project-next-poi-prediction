"""Dispatch the MNIST functionality to project.main.

The dispatch functions are used to
dynamically select the correct functions from the task
based on the hydra config file.
The following categories of functionality are grouped together:
    - train/test and fed test functions
    - net generator and dataloader generator functions
    - fit/eval config functions

The top-level project.dispatch
module operates as a pipeline
and selects the first function which does not return None.

Do not throw any errors based on not finding a given attribute
in the configs under any circumstances.

If you cannot match the config file,
return None and the dispatch of the next task
in the chain specified by project.dispatch will be used.
"""

from typing import Any

from omegaconf import DictConfig

from project.task.default.dispatch import (
    dispatch_config as dispatch_default_config,
    init_working_dir as init_working_dir_default,
)

from project.task.mcmg.dataset import get_dataloader_generators
from project.task.mcmg.models import get_mcmg_model
from project.task.mcmg.train_test import get_fed_eval_fn, test, train
from project.types.common import DataStructure, TrainStructure


def dispatch_train(
        cfg: DictConfig,
        **kwargs: Any,
) -> TrainStructure | None:
    """Dispatch the train/test and fed test functions based on the config file.

    Do not throw any errors based on not finding a given attribute
    in the configs under any circumstances.

    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the train function.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the train function.

    Returns
    -------
    Optional[TrainStructure]
        The train function, test function and the get_fed_eval_fn function.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with None default
    train_structure: str | None = cfg.get("task", {}).get(
        "train_structure",
        None,
    )

    # Only consider not None and uppercase matches
    if train_structure.upper() in ["CAL", "NY", "PHO", "SIN"]:
        return train, test, get_fed_eval_fn

    # Cannot match, send to next dispatch in chain
    return None


def dispatch_data(cfg: DictConfig, **kwargs: Any) -> DataStructure | None:
    """Dispatch the train/test and fed test functions based on the config file.

    Do not throw any errors based on not finding a given attribute
    in the configs under any circumstances.

    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the data functions.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the data functions.

    Returns
    -------
    Optional[DataStructure]
        The net generator, client dataloader generator and fed dataloader generator.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with {} default at nested dicts
    # and None default at the final key
    client_model_and_data: str | None = cfg.get(
        "task",
        {},
    ).get("model_and_data", None)

    data_dir: str | None = cfg.get("dataset", {}).get(
        "data_dir",
        None,
    )

    if client_model_and_data is not None and data_dir is not None:

        # Obtain the dataloader generators
        (client_dataloader_gen, fed_dataloader_gen) = get_dataloader_generators(data_dir, client_model_and_data.upper())

        if client_model_and_data.upper() in ["CAL", "NY", "PHO", "SIN"]:
            return (
                get_mcmg_model,
                client_dataloader_gen,
                fed_dataloader_gen,
                init_working_dir_default,
            )

        return None


dispatch_config = dispatch_default_config
