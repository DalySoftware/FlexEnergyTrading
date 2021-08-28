from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM  # type: ignore
from tensorflow.keras import Input  # type: ignore


class LstmAgent:
    @staticmethod
    def get_model(number_of_timesteps: int, number_of_features: int,
                  lstm_io_layer_units: int, lstm_compression_layer_units: int) -> Model:
        prediction_model = Sequential()

        prediction_model.add(LSTM(lstm_io_layer_units, return_sequences=True,
                                  batch_input_shape=(
                                      number_of_timesteps, 1, number_of_features
                                  )))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(
            LSTM(lstm_compression_layer_units, return_sequences=True))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(LSTM(lstm_io_layer_units))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(Dense(units=1))  # Prediction

        return prediction_model

    @staticmethod
    def get_model_2(number_of_timesteps: int, number_of_features: int,
                    lstm_io_layer_units: int, lstm_compression_layer_units: int) -> Model:
        prediction_model = Sequential()

        prediction_model.add(LSTM(lstm_io_layer_units, return_sequences=False,
                                  batch_input_shape=(
                                      number_of_timesteps, 1, number_of_features
                                  )))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(Dense(units=1))  # Prediction

        return prediction_model
