import logging
import math
from pathlib import Path
from typing import Literal

import pandas as pd
from flwr.common.logger import log
from omegaconf import OmegaConf


def greatest_power_of_two_at_most(n: int) -> int:
    return 2 ** math.floor(math.log2(n))

def generate_conf(
    is_federated: bool,
    heterogeneity: Literal["all_clients", "smaller_quantity_skew", "homogeneous_approximation"],
    city: Literal["CAL", "NY", "PHO", "SIN"],
    client_dropout_probability: float,
    seed: int,
    user_loc_counts: dict
):
    return {
        "defaults": [
            {
                "override /strategy": "fedavgflashback" if is_federated else "fedavg"
            }
        ],
        "task": {
            "dispatch_data": {
                "heterogeneity": heterogeneity,
                "city": city,
                "partition_type": "fed_natural" if is_federated else "centralised"
            },
            "fit_config": {
                "net_config": {
                    "h0_seed": seed,
                    "loc_count": user_loc_counts[heterogeneity][city]["loc_count"],
                    "user_count": 1 if is_federated else user_loc_counts[heterogeneity][city]["user_count"]
                },
                "dataloader_config": {
                    "loc_count": user_loc_counts[heterogeneity][city]["loc_count"],
                    "batch_size": 1 if is_federated else greatest_power_of_two_at_most(user_loc_counts[heterogeneity][city]["user_count"])
                },
                "run_config": {
                    "loc_count": user_loc_counts[heterogeneity][city]["loc_count"],
                    "batch_size": 1 if is_federated else greatest_power_of_two_at_most(user_loc_counts[heterogeneity][city]["user_count"]),
                    "client_dropout_probability": client_dropout_probability
                }
            },
            "eval_config": {
                "net_config": {
                    "h0_seed": seed,
                    "loc_count": user_loc_counts[heterogeneity][city]["loc_count"],
                    "user_count": 1 if is_federated else user_loc_counts[heterogeneity][city]["user_count"]
                },
                "dataloader_config": {
                    "loc_count": user_loc_counts[heterogeneity][city]["loc_count"],
                    "batch_size": 1 if is_federated else greatest_power_of_two_at_most(user_loc_counts[heterogeneity][city]["user_count"])
                },
                "run_config": {
                    "batch_size": 1 if is_federated else greatest_power_of_two_at_most(user_loc_counts[heterogeneity][city]["user_count"])
                }
            },
            "fed_test_config": {
                "net_config": {
                    "h0_seed": seed,
                    "loc_count": user_loc_counts[heterogeneity][city]["loc_count"],
                    "user_count": user_loc_counts[heterogeneity][city]["user_count"]
                },
                "dataloader_config": {
                    "loc_count": user_loc_counts[heterogeneity][city]["loc_count"],
                    "batch_size": greatest_power_of_two_at_most(user_loc_counts[heterogeneity][city]["user_count"])
                },
                "run_config": {
                    "batch_size": greatest_power_of_two_at_most(user_loc_counts[heterogeneity][city]["user_count"])
                }
            },
            "net_config_initial_parameters": {
                "h0_seed": seed,
                "loc_count": user_loc_counts[heterogeneity][city]["loc_count"],
                "user_count": user_loc_counts[heterogeneity][city]["user_count"]
            }
        },
        "fed": {
            "num_rounds": 20,
            "num_total_clients": user_loc_counts[heterogeneity][city]["user_count"] if is_federated else 1,
            "num_clients_per_round": math.ceil(user_loc_counts[heterogeneity][city]["user_count"] / 10) if is_federated else 1,
            "num_evaluate_clients_per_round": user_loc_counts[heterogeneity][city]["user_count"] if is_federated else 1
        }
    }

def generate_conf_nonfederated(
    city: Literal["CAL", "NY", "PHO", "SIN"],
    user_id: int,
    seed: int,
    user_loc_counts: dict
):
    return {
        "defaults": [
            {
                "override /strategy": "fedavg"
            }
        ],
        "task": {
            "dispatch_data": {
                "heterogeneity": "all_clients_nonfederated",
                "city": f"{city}-{user_id}",
                "partition_type": "centralised"
            },
            "fit_config": {
                "net_config": {
                    "h0_seed": seed,
                    "loc_count": user_loc_counts["all_clients"][city]["loc_count"],
                    "user_count": 1
                },
                "dataloader_config": {
                    "loc_count": user_loc_counts["all_clients"][city]["loc_count"],
                    "batch_size": 1
                },
                "run_config": {
                    "loc_count": user_loc_counts["all_clients"][city]["loc_count"],
                    "batch_size": 1,
                    "client_dropout_probability": 0.0
                }
            },
            "eval_config": {
                "net_config": {
                    "h0_seed": seed,
                    "loc_count": user_loc_counts["all_clients"][city]["loc_count"],
                    "user_count": 1
                },
                "dataloader_config": {
                    "loc_count": user_loc_counts["all_clients"][city]["loc_count"],
                    "batch_size": 1
                },
                "run_config": {
                    "batch_size": 1
                }
            },
            "fed_test_config": {
                "net_config": {
                    "h0_seed": seed,
                    "loc_count": user_loc_counts["all_clients"][city]["loc_count"],
                    "user_count": 1
                },
                "dataloader_config": {
                    "loc_count": user_loc_counts["all_clients"][city]["loc_count"],
                    "batch_size": 1
                },
                "run_config": {
                    "batch_size": 1
                }
            },
            "net_config_initial_parameters": {
                "h0_seed": seed,
                "loc_count": user_loc_counts["all_clients"][city]["loc_count"],
                "user_count": 1
            }
        },
        "fed": {
            "num_rounds": 20,
            "num_total_clients": 1,
            "num_clients_per_round": 1,
            "num_evaluate_clients_per_round": 1
        }
    }

IS_FEDERATED_TO_LABEL = {
    True: "federated",
    False: "centralised"
}

CITY_TO_LABEL = {
    "CAL": "cal",
    # "NY": "ny",
    "PHO": "pho",
    # "SIN": "sin"
}

HETEROGENEITY_TO_LABEL = {
    "all_clients": "ac",
    "smaller_quantity_skew": "sqs",
    "homogeneous_approximation": "ha"
}

CLIENT_DROPOUT_PROBABILITY_TO_LABEL = {
    0.0: "0p0",
    0.1: "0p1",
    0.15: "0p15",
    0.2: "0p2",
    0.25: "0p25"
}

SEEDS = (42, 361, 1337)

if __name__ == "__main__":
    YAMLS_DIR = Path("project") / "conf" / "task" / "flashback"
    YAMLS_DIR_NONFEDERATED = Path("project") / "conf" / "task" / "flashback-nonfederated"

    YAMLS_DIR.mkdir(parents=True, exist_ok=True)
    YAMLS_DIR_NONFEDERATED.mkdir(parents=True, exist_ok=True)

    user_loc_counts = {}
    for heterogeneity in HETEROGENEITY_TO_LABEL:
        user_loc_counts[heterogeneity] = {}
        for city in CITY_TO_LABEL:
            csv_path = Path("data") / "flashback" / "partition" / heterogeneity / city / "centralised" / "client_0.txt"
            df = pd.read_csv(csv_path, sep="\t", header=None)
            user_count = len(df.loc[:, 0].unique())
            loc_count = len(df.loc[:, 4].unique())
            user_loc_counts[heterogeneity][city] = {
                "user_count": user_count,
                "loc_count": loc_count
            }

    for is_federated, if_label in IS_FEDERATED_TO_LABEL.items():
        for city, c_label in CITY_TO_LABEL.items():
            for heterogeneity, h_label in HETEROGENEITY_TO_LABEL.items():
                for client_dropout_probability, cdp_label in CLIENT_DROPOUT_PROBABILITY_TO_LABEL.items():
                    for seed in SEEDS:
                        if not is_federated:
                            # If centralised, fix heterogeneity to all_clients and client_dropout_probability to 0
                            # as those only apply to the federated setting
                            if heterogeneity != "all_clients" or client_dropout_probability != 0.0:
                                continue
                        else:
                            # If federated and heterogeneity is not the default all_clients, fix client_dropout_probability to 0
                            # as investigations for hypotheses 2 and 3 are independent
                            # If federated and client_dropout_probability is not the default 0, fix heterogeneity to all_clients
                            # as investigations for hypotheses 2 and 3 are independent
                            if heterogeneity != "all_clients" and client_dropout_probability != 0.0:
                                continue

                        yaml_dict = generate_conf(is_federated, heterogeneity, city, client_dropout_probability, seed, user_loc_counts)
                        yaml_path = YAMLS_DIR / f"{if_label}-{c_label}-{h_label}-d{cdp_label}-s{seed}.yaml"
                        with yaml_path.open("w") as f:
                            f.writelines((
                                "# @package _global_\n",
                                "---\n"
                            ))
                            f.write(OmegaConf.to_yaml(yaml_dict))
                        log(logging.INFO, f"Wrote to {yaml_path}")

    # For single client training (non-federated setup)
    for city, c_label in CITY_TO_LABEL.items():
        for seed in SEEDS:
            for user_id in range(user_loc_counts["all_clients"][city]["user_count"]):
                yaml_dict = generate_conf_nonfederated(city, user_id, seed, user_loc_counts)
                yaml_path = YAMLS_DIR_NONFEDERATED / f"nonfederated-{c_label}{user_id}-s{seed}.yaml"
                with yaml_path.open("w") as f:
                    f.writelines((
                        "# @package _global_\n",
                        "---\n"
                    ))
                    f.write(OmegaConf.to_yaml(yaml_dict))
                log(logging.INFO, f"Wrote to {yaml_path}")
