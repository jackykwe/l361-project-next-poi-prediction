"""
A custom strategy, supporting custom management of User Embeddings in Flashback
between the federation's clients and server.
"""

import logging
from typing import cast

import flwr
import numpy as np
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate


# * Adapted from source code for FedAvg
class FedAvgFlashback(flwr.server.strategy.FedAvg):

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialize global model parameters."""
        # initial_parameters = self.initial_parameters
        # self.initial_parameters = None  # Don't keep initial parameters in memory
        # return initial_parameters
        # * We want to keep them in memory!
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        # * fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # * Return client/config pairs
        client_config_pairs = []
        for client in clients:
            # * same as parameters, but remove all other users' User Embeddings
            truncated_parameters_ndarrays = parameters_to_ndarrays(parameters)
            # * truncated_parameters_ndarrays above looks like
            # * [layer.shape for layer in truncated_parameters_ndarrays] =
            # * [(L, S), (U, S), (S, S), (S, S), (S,), (S,), (L, 2S), (L,)]
            # *  ^^^^^^  ******  $$$$$$$$$$$$$$$$$$$$$$$$$$  &&&&&&&&&&&&&
            # * where L is task.net_config_initial_parameters.loc_count (total number of locations in city dataset)
            # *       U is task.net_config_initial_parameters.user_count (total number of users in city dataset)
            # *       S is dataset.sequence_length
            # * and ^ represents the Location Embeddings
            # *     * represents the User Embeddings
            # *     $ represents the RNN
            # *     & represents the Linear layer
            truncated_parameters_ndarrays[1] = np.expand_dims(
                truncated_parameters_ndarrays[1][int(client.cid), :], axis=0
            )
            client_config_pairs.append((
                client,
                FitIns(ndarrays_to_parameters(truncated_parameters_ndarrays), config),
            ))
        log(
            logging.WARNING,
            f"configure_fit(): Providing truncated models for {[client.cid for client in clients]}",
        )
        return client_config_pairs
        # return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = []
        # * We remove User Embeddings from the weighted averaging step;
        # * we deal with those differently later
        client_cid_to_user_embeddings = {}
        for client, fit_res in results:
            user_embeddings_removed = parameters_to_ndarrays(fit_res.parameters)
            # * user_embeddings_removed above looks like
            # * [layer.shape for layer in user_embeddings_removed] =
            # * [(L, S), (1, S), (S, S), (S, S), (S,), (S,), (L, 2S), (L,)]
            # *  ^^^^^^  ******  $$$$$$$$$$$$$$$$$$$$$$$$$$  &&&&&&&&&&&&&
            # * and ^ represents the Location Embeddings
            # *     * represents the User Embeddings
            # *     $ represents the RNN
            # *     & represents the Linear layer
            client_cid_to_user_embeddings[int(client.cid)] = (
                user_embeddings_removed.pop(1)
            )
            weights_results.append((user_embeddings_removed, fit_res.num_examples))
        parameters_aggregated = aggregate(weights_results)

        # * Now we handle the User Embeddings
        # * Basically, each client is 100% responsible for their own user embedding
        # * and can only influence their own user embedding, not other clients'.

        # * We use initial_parameters_ndarrays as a state holder, to store the latest User Embeddings
        # * similar to how FedAvgM uses it to perform exponential averaging of gradients.
        # * Only parameters_to_ndarrays(self.initial_parameters)[1] is useful from this point in time onwards
        initial_parameters_ndarrays = parameters_to_ndarrays(
            cast(Parameters, self.initial_parameters)
        )

        # * Iterate over all users in the "global" dataset
        for client_cid in range(len(initial_parameters_ndarrays[1])):
            # * initial_parameters_ndarrays[1] has shape (U, S)
            # * where U is task.net_config_initial_parameters.user_count (total number of users in city dataset)
            if client_cid in client_cid_to_user_embeddings:
                initial_parameters_ndarrays[1][client_cid, :] = (
                    client_cid_to_user_embeddings[client_cid]
                )
        parameters_aggregated.insert(1, initial_parameters_ndarrays[1])  # type: ignore
        parameters_aggregated_result = ndarrays_to_parameters(parameters_aggregated)
        # * Save latest User Embeddings to self.initial_parameters for future rounds
        self.initial_parameters = ndarrays_to_parameters(initial_parameters_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        log(
            logging.WARNING,
            f"aggregate_fit(): Combining models from {[client.cid for client, _ in results]}",
        )
        return parameters_aggregated_result, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        # evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # * Same business as configure_fit() above
        # Return client/config pairs
        client_config_pairs = []
        for client in clients:
            # * same as parameters, but remove all other users' User Embeddings
            truncated_parameters_ndarrays = parameters_to_ndarrays(parameters)
            # * truncated_parameters_ndarrays above looks like
            # * [layer.shape for layer in truncated_parameters_ndarrays] =
            # * [(L, S), (U, S), (S, S), (S, S), (S,), (S,), (L, 2S), (L,)]
            # *  ^^^^^^  ******  $$$$$$$$$$$$$$$$$$$$$$$$$$  &&&&&&&&&&&&&
            # * where L is task.net_config_initial_parameters.loc_count (total number of locations in city dataset)
            # *       U is task.net_config_initial_parameters.user_count (total number of users in city dataset)
            # *       S is dataset.sequence_length
            # * and ^ represents the Location Embeddings
            # *     * represents the User Embeddings
            # *     $ represents the RNN
            # *     & represents the Linear layer
            truncated_parameters_ndarrays[1] = np.expand_dims(
                truncated_parameters_ndarrays[1][int(client.cid), :], axis=0
            )
            client_config_pairs.append((
                client,
                EvaluateIns(
                    ndarrays_to_parameters(truncated_parameters_ndarrays), config
                ),
            ))
        log(
            logging.WARNING,
            f"configure_evaluate(): Providing truncated models for {[client.cid for client in clients]}",
        )
        return client_config_pairs
