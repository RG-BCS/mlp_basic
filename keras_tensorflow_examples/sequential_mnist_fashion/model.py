# model.py

from tensorflow import keras

def build_sequential_model(input_shape, num_classes=10):
    model = keras.Sequential([
        keras.layers.InputLayer(shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
