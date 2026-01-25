from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class ClassificationAlgo(ABC):
    
    @abstractmethod
    def train(self, location_of_training_dataset):
        pass
    
    @abstractmethod
    def test(self, location_of_testing_dataset):
        pass
    
    @abstractmethod
    def get_weights(self):
        pass
    
    @abstractmethod
    def set_weights(self, weights):
        pass

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        X = df.drop(columns=['cardio'])
        y = df['cardio']
        return X.to_numpy(), y.to_numpy()

class XGBoostAlgo(ClassificationAlgo):
    def __init__(self):
        # Initialize XGBoost model
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def train(self, location_of_training_dataset):
        X_train, y_train = self.load_data(location_of_training_dataset)
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        self.model.fit(X_train, y_train)

    def test(self, location_of_testing_dataset):
        X_test, y_test = self.load_data(location_of_testing_dataset)
        print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def get_weights(self):
        return self.model.get_booster().get_fscore()

    def set_weights(self, weights):
        # XGBoost does not support direct weight setting
        pass

# Main execution
if __name__ == "__main__":
    xgb_algo = XGBoostAlgo()
    training_dataset_location = "./dataset/client1/training/train.csv"
    testing_dataset_location = "./dataset/client1/testing/test.csv"
    
    print("Training the XGBoost model...")
    xgb_algo.train(training_dataset_location)
    
    print("Testing the XGBoost model...")
    accuracy = xgb_algo.test(testing_dataset_location)
    print(f"Accuracy: {accuracy:.4f}")
