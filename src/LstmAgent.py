from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM  # type: ignore
from tensorflow.keras import Input  # type: ignore
from NeuralAgentHelper import NeuralAgentHelper
from TradeSignalHelper import TradeSignalHelper  # type: ignore

TRAINING_PERIOD_LENGTH = 1800
TRAINING_START_INDEX = 300
EPOCHS = 20
BATCH_SIZE = 300
IO_LAYER_UNITS = 100
COMPRESSION_LAYER_UNITS = 100
LOSS_STRING = "mean_squared_logarithmic_error"
SHUFFLE = True


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
    def run_evaluation(agent_type, training_period_length, eval_period_length, training_start_index, evaluation_start_index, epochs, batch_size,
                       io_layer_units, compression_layer_units, loss_string, use_shuffle):
        return NeuralAgentHelper.run_evaluation(agent_type, training_period_length, eval_period_length, training_start_index, evaluation_start_index, epochs, batch_size,
                                                io_layer_units, compression_layer_units, loss_string, use_shuffle)

    @staticmethod
    def get_trade_performance(eval_period_length, evaluation_start_index):
        y_predicted, close_prices, history = LstmAgent.run_evaluation(LstmAgent, TRAINING_PERIOD_LENGTH, eval_period_length, TRAINING_START_INDEX, evaluation_start_index,
                                                                      EPOCHS, BATCH_SIZE, IO_LAYER_UNITS, COMPRESSION_LAYER_UNITS, LOSS_STRING, SHUFFLE)

        y_predicted_vector = [y[0] for y in y_predicted]

        trade_volumes = TradeSignalHelper.get_trade_amounts_for_sigmoid(
            y_predicted_vector)

        hedged_volume, total_cost = TradeSignalHelper.get_trade_performance(
            close_prices, trade_volumes)

        return hedged_volume, total_cost
