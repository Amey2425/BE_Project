import os
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from dotenv import load_dotenv

from common.network import NeuralNetworkAlgo
from common.config import NUM_CLIENTS, ROUNDS, get_input_dim
from .evaluator import get_evaluate_fn

# Load Env
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# Setup paths for evaluation
GLOBAL_TEST = os.getenv("GLOBAL_TESTING_SET")
CLIENT_TESTS = {
    "Client1": os.getenv("CLIENT_1_TESTING_SET"),
    "Client2": os.getenv("CLIENT_2_TESTING_SET"),
    "Client3": os.getenv("CLIENT_3_TESTING_SET"),
}

def main():
    model = NeuralNetworkAlgo(input_dim=get_input_dim())
    initial_parameters = ndarrays_to_parameters(model.get_weights())

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(GLOBAL_TEST, CLIENT_TESTS),
        initial_parameters=initial_parameters,
    )

    # --- Capture the history object ---
    history = start_server(
        server_address="0.0.0.0:45678",
        config=ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )

    # --- FINAL SUMMARY REPORT ---
    print("\n" + "="*45)
    print(" FEDERATED LEARNING FINAL SUMMARY REPORT")
    print("="*45)

    # 1. Process Global Accuracy
    if "accuracy" in history.metrics_centralized:
        acc_list = history.metrics_centralized["accuracy"]
        # acc_list is [(round, acc), (round, acc), ...]
        final_round, final_acc = acc_list[-1]
        best_round, best_acc = max(acc_list, key=lambda x: x[1])

        print(f"Rounds Completed:   {final_round}")
        print(f"Final Global Acc:   {final_acc*100:.2f}%")
        print(f"Best Global Acc:    {best_acc*100:.2f}% (at Round {best_round})")
    
    # 2. Process Global Loss
    if history.losses_centralized:
        final_loss = history.losses_centralized[-1][1]
        print(f"Final Global Loss:  {final_loss:.4f}")

    print("="*45 + "\n")

if __name__ == "__main__":
    main()