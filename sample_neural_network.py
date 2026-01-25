import tensorflow as tf
import pandas as pd

class NeuralNetworkAlgo():
    def __init__(self):
        # Define a custom learning rate
        learning_rate = 0.005
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),  # 5 input features
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile the model with the custom optimizer
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self, location_of_training_dataset):
        X_train, y_train = self.load_data(location_of_training_dataset)
        self.model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    def test(self, location_of_testing_dataset):
        # Load test data
        X_test, y_test = self.load_data(location_of_testing_dataset)
        
        # Evaluate the model on test data; evaluate returns [loss, accuracy]
        loss, accuracy = self.model.evaluate(X_test, y_test)
        
        # Return both loss and accuracy
        return loss, accuracy
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

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

# Main execution
if __name__ == "__main__":
    # Instantiate the neural network algorithm
    nn_algo = NeuralNetworkAlgo()
    
    # Define the paths to your training and testing datasets
    training_dataset_location = "./dataset/client1/training/train.csv"
    testing_dataset_location = "./dataset/client1/testing/test.csv"
    
    # Train the model
    print("Training the model...")
    nn_algo.train(training_dataset_location)
    
    # Test the model
    print("Testing the model...")
    loss, accuracy = nn_algo.test(testing_dataset_location)
    
    # Output the results
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

'''
Training the model...
Epoch 1/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 1s 761us/step - accuracy: 0.4998 - loss: 183.8534  
Epoch 2/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 0s 752us/step - accuracy: 0.5105 - loss: 29.8005
Epoch 3/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 0s 714us/step - accuracy: 0.5176 - loss: 19.0381
Epoch 4/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 0s 724us/step - accuracy: 0.5165 - loss: 9.3997 
Epoch 5/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 0s 714us/step - accuracy: 0.5103 - loss: 5.9072 
Epoch 6/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 0s 718us/step - accuracy: 0.5199 - loss: 4.1477
Epoch 7/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 0s 716us/step - accuracy: 0.5218 - loss: 2.6872
Epoch 8/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 0s 745us/step - accuracy: 0.5235 - loss: 2.5575
Epoch 9/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 0s 706us/step - accuracy: 0.5238 - loss: 2.1188
Epoch 10/10
456/456 ━━━━━━━━━━━━━━━━━━━━ 0s 724us/step - accuracy: 0.5520 - loss: 1.2250
Testing the model...
114/114 ━━━━━━━━━━━━━━━━━━━━ 0s 636us/step - accuracy: 0.5063 - loss: 1.6202
Loss: 1.6545, Accuracy: 0.4985
'''