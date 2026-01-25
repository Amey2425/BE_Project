from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

class RandomForestAlgo(ClassificationAlgo):
    def __init__(self):
        # Initialize Random Forest model
        self.model = RandomForestClassifier(n_estimators=100)

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
        return self.model.feature_importances_

    def set_weights(self, weights):
        # RandomForest does not allow setting weights directly
        # This is left blank, as weights are determined by the ensemble
        pass

# Main execution
if __name__ == "__main__":
    rf_algo = RandomForestAlgo()
    training_dataset_location = "./dataset/client1/training/train.csv"
    testing_dataset_location = "./dataset/client1/testing/test.csv"
    
    print("Training the Random Forest model...")
    rf_algo.train(training_dataset_location)
    
    print("Testing the Random Forest model...")
    accuracy = rf_algo.test(testing_dataset_location)
    print(f"Accuracy: {accuracy:.4f}")
