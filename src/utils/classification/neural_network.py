# from .classification import ClassificationAlgo
# import tensorflow as tf

# class NeuralNetworkAlgo(ClassificationAlgo):
#     def __init__(self):
#         # Define a custom learning rate
#         learning_rate = 0.005
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
#         self.model = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=(4,)),
#             tf.keras.layers.Dense(64, activation='relu'),  
#             tf.keras.layers.Dense(32, activation='relu'),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
        
#         self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
#     def train(self, location_of_training_dataset):
#         X_train, y_train = self.load_data(location_of_training_dataset)
#         self.model.fit(X_train, y_train, epochs=10, batch_size=32)
    
#     def test(self, location_of_testing_dataset):
#         X_test, y_test = self.load_data(location_of_testing_dataset)
        
#         loss, accuracy = self.model.evaluate(X_test, y_test)
        
#         return loss, accuracy
    
#     def get_weights(self):
#         return self.model.get_weights()
    
#     def set_weights(self, weights):
#         self.model.set_weights(weights)
from .classification import ClassificationAlgo
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import pandas as pd
import pickle
import numpy as np

# Load scaler and column order globally (for speed)
with open("./preprocessing/scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

with open("./preprocessing/columns.txt") as f:
    COL_ORDER = [line.strip() for line in f.readlines()]


class NeuralNetworkAlgo(ClassificationAlgo):

    def __init__(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.model = tf.keras.Sequential([
            layers.Input(shape=(len(COL_ORDER),)),
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

        self.model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])


    # ---------------------------
    # LOADING + SCALING DATA
    # ---------------------------
    def load_data(self, path):
        df = pd.read_csv(path)

        # Add age_years back if present
        if "age_years" not in df.columns:
            df["age_years"] = df["age"] / 365

        y = df["cardio"].values
        X = df.drop(columns=["cardio"])

        # Enforce global column order
        X = X[COL_ORDER]

        # Apply scaler
        X = SCALER.transform(X)

        return X.astype(np.float32), y.astype(np.float32)

    # ---------------------------
    # TRAIN
    # ---------------------------
    def train(self, path):
        X_train, y_train = self.load_data(path)

        callback = tf.keras.callbacks.EarlyStopping(
            patience=4,
            monitor="loss",
            restore_best_weights=True
        )

        class_weight = {0: 1.2, 1: 1.0}

        self.model.fit(
            X_train, y_train,
            epochs=35,
            batch_size=64,
            verbose=0,
            callbacks=[callback],
            class_weight=class_weight
        )

    # ---------------------------
    def test(self, path):
        X_test, y_test = self.load_data(path)
        return self.model.evaluate(X_test, y_test, verbose=0)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
