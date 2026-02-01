import tensorflow as tf
from common.network import NeuralNetworkAlgo
from .data_loader import load_local_data

class ClientTrainer(NeuralNetworkAlgo):
    """Extends the common architecture with local training logic."""
    
    def train(self, path, epochs=35):
        x, y = load_local_data(path)
        callback = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
        
        self.model.fit(
            x, y, 
            epochs=epochs, 
            batch_size=64, 
            verbose=0, 
            callbacks=[callback],
            class_weight={0: 1.2, 1: 1.0}
        )

    def personalize(self, path, epochs=3):
        """Per-FedAVG style personalization: fine-tune only the last layer."""
        x, y = load_local_data(path)
        
        # Freeze all layers except the last Dense layer
        for layer in self.model.layers[:-1]:
            layer.trainable = False
        
        self.model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
        
        # Unfreeze for next global round
        for layer in self.model.layers:
            layer.trainable = True

    def test(self, path):
        x, y = load_local_data(path)
        return self.model.evaluate(x, y, verbose=0)