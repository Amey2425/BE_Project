import os
import argparse
from flwr.client import NumPyClient, start_client
from dotenv import load_dotenv

# Import your local modules
from .local_trainer import ClientTrainer
from .data_loader import load_local_data

# 1. Define the class (keeping your structure)
class FlowerClient(NumPyClient):
    def __init__(self, client_id, train_path, test_path, use_personalization=False):
        # Ensure input_dim matches your columns.txt count
        self.trainer = ClientTrainer(input_dim=12) 
        self.train_path = train_path
        self.test_path = test_path
        self.use_personalization = use_personalization

    def get_parameters(self, config):
        return self.trainer.get_weights()

    def fit(self, parameters, config):
        self.trainer.set_weights(parameters)
        self.trainer.train(self.train_path)
        if self.use_personalization:
            self.trainer.personalize(self.train_path)
            
        _, y = load_local_data(self.train_path)
        return self.trainer.get_weights(), len(y), {}

    def evaluate(self, parameters, config):
        self.trainer.set_weights(parameters)
        loss, acc = self.trainer.test(self.test_path)
        _, y = load_local_data(self.test_path)
        return float(loss), len(y), {"accuracy": float(acc)}

# 2. Execution Logic
if __name__ == "__main__":
    # Load environment variables (datasets paths)
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--personalize", action="store_true")
    args = parser.parse_args()

    # Get paths from .env using the ID passed by main.py
    train_set = os.getenv(f"CLIENT_{args.id}_TRAINING_SET")
    test_set = os.getenv(f"CLIENT_{args.id}_TESTING_SET")

    if not train_set or not test_set:
        print(f"❌ Error: Paths for Client {args.id} not found in .env")
        exit(1)

    # 3. INSTANTIATE AND START
    print(f"📡 Launching Client {args.id} (Personalization: {args.personalize})...")
    
    client_instance = FlowerClient(
        client_id=args.id,
        train_path=train_set,
        test_path=test_set,
        use_personalization=args.personalize
    )

    start_client(
        server_address="127.0.0.1:45678", 
        client=client_instance.to_client() # Use to_client() for newer Flower versions
    )