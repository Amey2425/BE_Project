# from utils.classification.neural_network import NeuralNetworkAlgo
# import os
# from flwr.client import NumPyClient, start_client
# from flwr.common import NDArrays, Scalar
# from typing import Dict

# from dotenv import load_dotenv
# # Load .env from parent directory of src/
# env_path = os.path.join(os.path.dirname(__file__), '../.env')
# load_dotenv(env_path)

# CLIENT_TRAINING_SET = os.getenv('CLIENT_2_TRAINING_SET')
# CLIENT_TESTING_SET = os.getenv('CLIENT_2_TESTING_SET')
# GLOBAL_TESTING_SET = os.getenv('GLOBAL_TESTING_SET')
# CLIENT_NAME = 'client2'


# class FlowerClient(NumPyClient):
#     def __init__(self, model, trainset_location, testset_location):
#         self.model = model
#         self.trainset_location = trainset_location
#         self.testset_location = testset_location
#         self.round_number = 0

#     # Train the model
#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)
#         self.model.train(self.trainset_location)
#         num_samples = len(self.model.load_data(self.trainset_location)[1])
#         self.round_number += 1
#         return self.model.get_weights(), num_samples, {}

#     # Test the model
#     def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
#         # Update model parameters
#         self.model.set_weights(parameters)

#         # Perform evaluation (assumes your model has a test method that returns loss and accuracy)
#         loss, accuracy = self.model.test(self.testset_location)

#         # Calculate the number of samples in the test set
#         num_samples = len(self.model.load_data(self.testset_location)[1])

#         # Return the required tuple (loss, num_samples, and metrics dictionary)
#         return loss, num_samples, {"accuracy": accuracy}


# model = NeuralNetworkAlgo()
# start_client(server_address="127.0.0.1:8000", client=FlowerClient(model, CLIENT_TRAINING_SET, CLIENT_TESTING_SET).to_client())    
import os
from flwr.client import NumPyClient, start_client
from dotenv import load_dotenv
from utils.classification.neural_network import NeuralNetworkAlgo

env_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(env_path)

TRAIN = os.getenv("CLIENT_2_TRAINING_SET")
TEST = os.getenv("CLIENT_2_TESTING_SET")


class FlowerClient(NumPyClient):
    def __init__(self, model, trainset_location, testset_location):
        self.model = model
        self.trainset_location = trainset_location
        self.testset_location = testset_location

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config=None):
        # Set global weights from server
        self.model.set_weights(parameters)

        # ---------- FEDERATED TRAINING ----------
        self.model.train(self.trainset_location)

        # ---------- PER-FEDAVG PERSONALIZATION ----------
        # Fine-tune only last layer
        X_train, y_train = self.model.load_data(self.trainset_location)

        # Freeze all layers except last Dense(1)
        for layer in self.model.model.layers[:-1]:
            layer.trainable = False

        # Train last layer for personalization
        self.model.model.fit(
            X_train, y_train,
            epochs=3,
            batch_size=32,
            verbose=0
        )

        # Return updated personalized weights
        _, y = self.model.load_data(self.trainset_location)
        return self.model.get_weights(), len(y), {}

    def evaluate(self, parameters, config=None):
        self.model.set_weights(parameters)
        loss, acc = self.model.test(self.testset_location)
        _, y = self.model.load_data(self.testset_location)
        return loss, len(y), {"accuracy": acc}


if __name__ == "__main__":
    model = NeuralNetworkAlgo()
    client = FlowerClient(model, TRAIN, TEST)
    start_client(server_address="127.0.0.1:8000", client=client)
