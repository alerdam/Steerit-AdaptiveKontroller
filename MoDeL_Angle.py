# MoDeL_Angle.py

import tensorflow as tf
from tensorflow import keras
import numpy as np
import handlib

def MODEL_network():
    model = keras.models.Sequential([
        keras.layers.Input(shape=(8,)), 
        keras.layers.Dense(8, activation='tanh'),   
        keras.layers.Dense(16, activation='sigmoid'), 
        keras.layers.Dense(8, activation='sigmoid'),
        keras.layers.Dense(units=1, activation='tanh') 
    ])
    optmzr = keras.optimizers.Adam(learning_rate=0.01)
    lossfunc = keras.losses.MeanAbsoluteError()
    model.compile(optimizer=optmzr, loss=lossfunc, metrics=["mae"])
    return model

# Build model
model = MODEL_network()
model.summary()

# Load data
loaded_data = handlib.load_csv_to_numpy("DATA/DATABASE.csv")
if loaded_data is not None:
    xtrain, ytrain, xtest, ytest = handlib.Explore_DS(loaded_data)
    if xtrain is not None:
        print(" -------- DATA --------")
        print("xtrain shape:", xtrain.shape)
        print("ytrain shape:", ytrain.shape)
        print("xtest shape:", xtest.shape)
        print("ytest shape:", ytest.shape)
        print("----------------------------------------------")

        # Train
        model.fit(xtrain, ytrain, batch_size=8, epochs=50)

        # Evaluate
        print("\n----------------------------------------------")
        print("TEST RESULTS : Model Angle")
        model.evaluate(xtest, ytest)

        # Save
        model.save("models/M_Angle.keras")
        print("Model created and saved as M_Angle.keras")