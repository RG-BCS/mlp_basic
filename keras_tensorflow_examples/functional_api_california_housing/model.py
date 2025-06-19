# model.py

from tensorflow import keras

def build_functional_model(input_shape_a, input_shape_b):
    input_A = keras.layers.Input(shape=input_shape_a, name='wide_input')
    input_B = keras.layers.Input(shape=input_shape_b, name='deep_input')

    hidden1 = keras.layers.Dense(30, activation='relu')(input_B)
    hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
    concat = keras.layers.Concatenate()([input_A, hidden2])

    output = keras.layers.Dense(1, name='output')(concat)

    model = keras.Model(inputs=[input_A, input_B], outputs=[output])
    return model
