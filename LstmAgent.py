from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM  # type: ignore
from tensorflow.keras import Input  # type: ignore


class LstmAgent:
    @staticmethod
    def get_model(number_of_timesteps: int, number_of_features: int) -> Model:
        prediction_model = Sequential()

        LSTM_IO_LAYER_UNITS = 100
        LSTM_COMPRESSION_LAYER_UNITS = 20

        prediction_model.add(LSTM(LSTM_IO_LAYER_UNITS, return_sequences=True, batch_input_shape=(
            number_of_timesteps, 1, number_of_features)))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(
            LSTM(LSTM_COMPRESSION_LAYER_UNITS, return_sequences=True))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(LSTM(LSTM_IO_LAYER_UNITS))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(Dense(units=1))  # Prediction

        # prediction_model.summary()
        return prediction_model
