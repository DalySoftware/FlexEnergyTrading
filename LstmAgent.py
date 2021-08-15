from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM  # type: ignore
from tensorflow.keras import Input  # type: ignore


class LstmAgent:
    @staticmethod
    def get_model(number_of_timesteps: int, number_of_features: int) -> Model:
        prediction_model = Sequential()

        LSTM_UNITS_PER_LAYER = 500

        prediction_model.add(LSTM(LSTM_UNITS_PER_LAYER, return_sequences=True, batch_input_shape=(
            number_of_timesteps, 1, number_of_features)))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(
            LSTM(LSTM_UNITS_PER_LAYER, return_sequences=True))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(
            LSTM(LSTM_UNITS_PER_LAYER, return_sequences=True))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(LSTM(LSTM_UNITS_PER_LAYER))
        prediction_model.add(Dropout(0.2))
        prediction_model.add(Dense(units=1))  # Prediction

        # prediction_model.summary()
        return prediction_model
