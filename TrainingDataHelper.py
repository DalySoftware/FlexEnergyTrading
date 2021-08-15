from typing import NamedTuple, Tuple
from LstmAgent import LstmAgent
import sys
from pandas.core.series import Series
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from NaturalGasDataProvider import NaturalGasDataProvider
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt  # type: ignore


class TrainingDataHelper:
    @staticmethod
    def scale_minus_one_to_one(data_frame: DataFrame, column: str):
        result = data_frame.copy()

        min_max_scaler = MinMaxScaler()

        result[column] = min_max_scaler.fit_transform(
            result[column].values.reshape(-1, 1)
        )

        return result

    @classmethod
    def normalize_data(cls, data_frame: DataFrame) -> DataFrame:
        result = data_frame.copy()

        result = cls.scale_minus_one_to_one(result, "Open")
        result = cls.scale_minus_one_to_one(result, "Close")
        result = cls.scale_minus_one_to_one(result, "High")
        result = cls.scale_minus_one_to_one(result, "Low")
        result = cls.scale_minus_one_to_one(result, "Volume")

        return result

    @staticmethod
    def train_model(scaled_training_data: DataFrame):
        pass

    @staticmethod
    def get_training_data(scaled, training_start_index, training_length):
        x_training: DataFrame = scaled[training_start_index:training_start_index +
                                       training_length].copy()
        x_training["DaysLeftInWindow"] = np.flip(np.arange(training_length))
        x_training["RunningMinClose"] = x_training["Close"].expanding().min()

        x_copy = x_training.copy()
        x_copy["GlobalMin"] = x_copy["Close"].min()

        x_copy["DiffFromGlobalMin"] = x_copy.loc[:, ["Close", "GlobalMin"]].apply(
            lambda row: row["Close"] - row["GlobalMin"], axis=1)

        y_training: DataFrame = x_copy.loc[:, ["DiffFromGlobalMin"]]

        return x_training, y_training


if __name__ == '__main__':
    data = NaturalGasDataProvider.get_data()

    scaled = TrainingDataHelper.normalize_data(data)

    training_start_index = 100
    trading_period_length = 100

    x_training, y_training = TrainingDataHelper.get_training_data(
        scaled, training_start_index, trading_period_length)

    x_training_np = np.array(x_training)
    y_training_np = np.array(y_training)

    assert x_training_np.shape[0] == trading_period_length

    x_training_np_reshaped = np.reshape(
        x_training_np, (x_training_np.shape[0], 1, x_training_np.shape[1]))

    model = LstmAgent.get_model(trading_period_length, x_training_np.shape[1])

    EPOCHS = 100
    BATCH_SIZE = trading_period_length

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.summary()

    print(f"x_training_np_reshaped.shape: {x_training_np_reshaped.shape}")
    print(f"y_training_np.shape: {y_training_np.shape}")

    model.fit(x_training_np_reshaped, y_training_np,
              epochs=EPOCHS, batch_size=BATCH_SIZE)

    x_testing, y_testing = TrainingDataHelper.get_training_data(
        scaled, training_start_index + trading_period_length, trading_period_length)

    x_testing_np = np.array(x_testing)
    x_testing_np_reshaped = np.reshape(
        x_testing_np, (x_testing_np.shape[0], 1, x_testing_np.shape[1]))

    print(x_testing_np_reshaped.shape)
    y_predicted_on_testing_set = model.predict(
        x_testing_np_reshaped, batch_size=BATCH_SIZE)

    fig = plt.figure()
    x_values_for_plot = range(1, trading_period_length + 1)

    training = fig.add_subplot(1, 2, 1)
    testing = fig.add_subplot(1, 2, 2)

    y_predicted_on_training_set = model.predict(
        x_training_np_reshaped, batch_size=BATCH_SIZE)
    #types: ignore
    training.plot(x_values_for_plot, y_training,
                  x_values_for_plot, y_predicted_on_training_set)

    training.set_title("Training Data")

    #types: ignore
    testing.plot(x_values_for_plot, y_testing,
                 x_values_for_plot, y_predicted_on_testing_set)
    testing.set_title("Testing Data")

    plt.show()
