from logging import INFO
from flwr.common.logger import log
from common.network import NeuralNetworkAlgo

def get_evaluate_fn(global_test_path, client_paths):
    def evaluate(server_round, parameters, config):
        model = NeuralNetworkAlgo(input_dim=12) 
        model.set_weights(parameters)

        # 1. Global Evaluation - Capture BOTH loss and accuracy
        loss_global, acc_global = model.test(global_test_path)
        
        # 2. Structured Logging for better visibility during the run
        log(INFO, f"\n" + "-"*40)
        log(INFO, f"🌐 ROUND {server_round} GLOBAL EVALUATION")
        log(INFO, f"   Acc: {acc_global:.4f} | Loss: {loss_global:.4f}")
        
        for name, path in client_paths.items():
            _, acc = model.test(path)
            log(INFO, f"   {name} Test Acc: {acc:.4f}")
        log(INFO, "-"*40 + "\n")

        # Return actual loss so the 'centralized loss' history isn't 0.0
        return float(loss_global), {"accuracy": float(acc_global)}
    
    return evaluate