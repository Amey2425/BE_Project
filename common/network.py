import tensorflow as tf
from tensorflow.keras import layers, regularizers
from .classification_base import ClassificationAlgo

class NeuralNetworkAlgo(ClassificationAlgo):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()

    def _build_model(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model = tf.keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.BatchNormalization(),

            layers.Dense(256, activation="relu", kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.20),

            layers.Dense(128, activation="relu", kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.15),

            layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            layers.Dropout(0.10),

            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def test(self, path):
        """
        Added this back to common so the Server can evaluate!
        """
        if not path:
            return 0.0, 0.0
            
        # Import inside the method to avoid circular imports if necessary
        from client.data_loader import load_local_data
        
        x, y = load_local_data(path)
        # model.evaluate returns [loss, accuracy]
        results = self.model.evaluate(x, y, verbose=0)
        return results[0], results[1] # Returns (loss, accuracy)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train(self, path):
        pass