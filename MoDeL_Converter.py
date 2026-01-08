import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("models\M_Angle.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("models/M_Angle.tflite", "wb") as f:
    f.write(tflite_model)