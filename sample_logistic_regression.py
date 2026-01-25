from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Separate features (X) and target (y)
        X = df.drop(columns=['cardio'])  # Drop the target column to get features
        y = df['cardio']  # Target column
        
        # Convert to NumPy arrays
        X = X.to_numpy()
        y = y.to_numpy()
        
        return X, y

class LogisticRegressionAlgo(ClassificationAlgo):
    def __init__(self):
        # Initialize the logistic regression model
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)

    def train(self, location_of_training_dataset):
        X_train, y_train = self.load_data(location_of_training_dataset)
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")  # Debugging line
        self.model.fit(X_train, y_train)

    def test(self, location_of_testing_dataset):
        # Load test data
        X_test, y_test = self.load_data(location_of_testing_dataset)
        print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")  # Debugging line
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def get_weights(self):
        return self.model.coef_, self.model.intercept_

    def set_weights(self, weights):
        # Set weights; weights should be a tuple of (coef, intercept)
        coef, intercept = weights
        self.model.coef_ = coef
        self.model.intercept_ = intercept

# Main execution
if __name__ == "__main__":
    # Instantiate the logistic regression algorithm
    lr_algo = LogisticRegressionAlgo()
    
    # Define the paths to your training and testing datasets
    training_dataset_location = "./dataset/client1/training/train.csv"
    testing_dataset_location = "./dataset/client1/testing/test.csv"
        
    # Train the model
    print("Training the model...")
    lr_algo.train(training_dataset_location)
    
    # Test the model
    print("Testing the model...")
    accuracy = lr_algo.test(testing_dataset_location)
    
    # Output the results
    print(f"Accuracy: {accuracy:.4f}")
