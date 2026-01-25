# from flwr.common import ndarrays_to_parameters
# from flwr.server import ServerConfig, start_server
# from flwr.server.strategy import FedAvg, FedMedian, FedAdam, FedProx
# from flwr.common.logger import console_handler, log
# from logging import INFO, ERROR
# import os
# from dotenv import load_dotenv

# # Load .env from parent directory of src/
# env_path = os.path.join(os.path.dirname(__file__), '../.env')
# load_dotenv(env_path)

# from utils.classification.neural_network import NeuralNetworkAlgo

# GLOBAL_TESTING_SET = os.getenv('GLOBAL_TESTING_SET')
# CLIENT_1_TESTING_SET = os.getenv('CLIENT_1_TESTING_SET')
# CLIENT_2_TESTING_SET = os.getenv('CLIENT_2_TESTING_SET')
# CLIENT_3_TESTING_SET = os.getenv('CLIENT_3_TESTING_SET')


# def evaluate(server_round, parameters, config):
#     model = NeuralNetworkAlgo();
#     model.set_weights(parameters)

#     _,accuracy_global = model.test(GLOBAL_TESTING_SET)
#     _,accuracy_client1 = model.test(CLIENT_1_TESTING_SET)
#     _,accuracy_client2 = model.test(CLIENT_2_TESTING_SET)
#     _,accuracy_client3 = model.test(CLIENT_3_TESTING_SET)

#     log(INFO, "test accuracy on global testset: %.4f", accuracy_global)
#     log(INFO, "test accuracy on client1 testset: %.4f", accuracy_client1)
#     log(INFO, "test accuracy on client2 testset: %.4f", accuracy_client2)
#     log(INFO, "test accuracy on client3 testset: %.4f", accuracy_client3)


# model = NeuralNetworkAlgo()
# params = ndarrays_to_parameters(model.get_weights())


# # Start Flower server
# start_server(
#   server_address="0.0.0.0:8000",
#   config=ServerConfig(num_rounds=10),
#   strategy=FedProx(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         initial_parameters=params,
#         evaluate_fn=evaluate,
#         min_fit_clients=3,      
#         min_evaluate_clients=3,  
#         min_available_clients=3,
#         proximal_mu=0.03
#     )
# )

# ===============================
# Flower Federated Learning Server
# Fully Fixed for Flower 1.10.0
# ===============================
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from flwr.common.logger import log
from logging import INFO

import os
from dotenv import load_dotenv

from utils.classification.neural_network import NeuralNetworkAlgo

env_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(env_path)

GLOBAL_TEST = os.getenv("GLOBAL_TESTING_SET")
CLIENT1_TEST = os.getenv("CLIENT_1_TESTING_SET")
CLIENT2_TEST = os.getenv("CLIENT_2_TESTING_SET")
CLIENT3_TEST = os.getenv("CLIENT_3_TESTING_SET")


def evaluate(server_round, parameters, _):
    model = NeuralNetworkAlgo()
    model.set_weights(parameters)

    _, acc_global = model.test(GLOBAL_TEST)
    _, acc_c1 = model.test(CLIENT1_TEST)
    _, acc_c2 = model.test(CLIENT2_TEST)
    _, acc_c3 = model.test(CLIENT3_TEST)

    log(INFO, f"[Round {server_round}] Global Acc: {acc_global:.4f}")
    log(INFO, f"Client1: {acc_c1:.4f}, Client2: {acc_c2:.4f}, Client3: {acc_c3:.4f}")

    return 0.0, {"accuracy": acc_global}


model = NeuralNetworkAlgo()
initial_parameters = ndarrays_to_parameters(model.get_weights())

strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_available_clients=3,
    min_evaluate_clients=3,
    evaluate_fn=evaluate,
    initial_parameters=initial_parameters,
)

start_server(
    server_address="0.0.0.0:8000",
    config=ServerConfig(num_rounds=50),
    strategy=strategy
)
