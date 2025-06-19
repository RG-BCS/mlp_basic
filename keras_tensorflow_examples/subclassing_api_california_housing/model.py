# model.py

from tensorflow import keras

class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
        self.concat = keras.layers.Concatenate()

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = self.concat([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

    def custom_evaluate_(self, x=None, y=None, batch_size=None, verbose=1,
                         sample_weight=None, steps=None, callbacks=None, max_queue_size=10,
                         workers=1, use_multiprocessing=False):
        y_pred = self.predict(x, verbose=verbose)
        main_output, aux_output = y_pred
        loss_fn = keras.losses.MeanSquaredError()
        main_loss = loss_fn(y[0], main_output).numpy()
        aux_loss = loss_fn(y[1], aux_output).numpy()
        total_loss = 0.9 * main_loss + 0.1 * aux_loss
        if verbose:
            print(f"Main loss: {main_loss}, Aux loss: {aux_loss}, Total loss: {total_loss}")
        return total_loss
