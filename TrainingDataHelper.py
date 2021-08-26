from tensorflow.python.keras.engine import training
from LstmAgent import LstmAgent
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

        # type: ignore
        if (np.any(np.isnan(result[column]))):
            result[column] = np.zeros(result[column].shape)

        # type: ignore
        if (not np.all(np.isfinite(result[column]))):
            result[column] = np.zeros(result[column].shape)

        result[column] = min_max_scaler.fit_transform(
            result[column].values.reshape(-1, 1)
        )

        return (result, min_max_scaler)

    @classmethod
    def normalize_data(cls, data_frame: DataFrame) -> DataFrame:
        result = data_frame.copy()

        for key in result.keys():  # type: ignore
            result, _ = cls.scale_minus_one_to_one(result, key)

        # result, _ = cls.scale_minus_one_to_one(result, "Open")
        # result, _ = cls.scale_minus_one_to_one(result, "Close")
        # result, _ = cls.scale_minus_one_to_one(result, "High")
        # result, _ = cls.scale_minus_one_to_one(result, "Low")
        # result, _ = cls.scale_minus_one_to_one(result, "Volume")

        return result

    @staticmethod
    def get_x_y_data(scaled, start_index, length, min_scaler):
        x: DataFrame = scaled[start_index:start_index + length].copy()

        x["DaysLeftInWindow"] = np.flip(np.arange(length))
        x["RunningMinClose"] = x["Close"].expanding().min()

        x_copy = x.copy()
        x_copy["GlobalMin"] = x_copy["Close"].min()

        x_copy["DiffFromGlobalMin"] = x_copy.loc[:, ["Close", "GlobalMin"]].apply(
            lambda row: row["Close"] - row["GlobalMin"], axis=1)

        x_copy["DiffFromGlobalMinScaled"] = min_scaler.fit_transform(
            x_copy["DiffFromGlobalMin"].values.reshape(-1, 1))

        # x_copy["CloseScaled"] = min_scaler.fit_transform(
        #     x_copy["Close"].values.reshape(-1, 1))

        y: DataFrame = x_copy.loc[:, ["DiffFromGlobalMinScaled"]]

        return x, y

    @staticmethod
    def plot_results(TRAINING_START_INDEX, TRADING_PERIOD_LENGTH,
                     BATCH_SIZE, EPOCHS,
                     x_training_np_reshaped, y_training,
                     y_testing, y_predicted_on_testing_set,
                     model, loss_string,  use_shuffle, fit_history):

        fig = plt.figure()
        x_values_for_plot = range(1, TRADING_PERIOD_LENGTH + 1)

        training_plot = fig.add_subplot(2, 2, 1)
        testing_plot = fig.add_subplot(2, 2, 2)
        convergence_plot = fig.add_subplot(2, 2, 3)
        text_plot = fig.add_subplot(2, 2, 4)

        y_predicted_on_training_set = model.predict(
            x_training_np_reshaped, batch_size=BATCH_SIZE
        )
        #types: ignore
        training_plot.plot(x_values_for_plot, y_training,
                           x_values_for_plot, y_predicted_on_training_set)
        training_plot.set_xlabel("Epoch")
        training_plot.set_title("Training Data")
        training_plot.legend(["Training", "Predicted"], loc="upper left")

        #types: ignore
        testing_plot.plot(x_values_for_plot, y_testing,
                          x_values_for_plot, y_predicted_on_testing_set)
        testing_plot.set_xlabel("Epoch")
        testing_plot.set_title("Testing Data")
        training_plot.legend(["Testing", "Predicted"], loc="upper left")

        #types: ignore
        convergence_plot.plot(fit_history.history["loss"])
        convergence_plot.plot(fit_history.history["val_loss"])
        convergence_plot.set_title("Loss")
        convergence_plot.set_ylabel("Loss")
        convergence_plot.set_xlabel("Epoch")
        convergence_plot.legend(["Train", "Test"], loc="upper left")

        info_text = f"Epochs: {EPOCHS}  " + \
                    f"\nTraining Start Index: {TRAINING_START_INDEX}  " + \
                    f"\nTrading Period Length: {TRADING_PERIOD_LENGTH}" + \
                    f"\nBatch Size: {BATCH_SIZE}" + \
                    f"\nLoss Function: {loss_string}" + \
                    f"\nShuffle: {use_shuffle}"

        text_plot.text(0.5, 0.5, info_text,
                       horizontalalignment='left', verticalalignment='center',
                       )
        text_plot.set_title("Parameters")

        plt.show()


def main():
    data = NaturalGasDataProvider.get_data(True)

    scaled = TrainingDataHelper.normalize_data(data)

    # print(scaled[200:].head())
    # print(scaled[200:].tail())
    # scaled.to_csv("C:\\temp\\temp.csv")

    TRAINING_START_INDEX = 300
    TRADING_PERIOD_LENGTH = 300

    y_training_scaler = MinMaxScaler()

    x_training, y_training = TrainingDataHelper.get_x_y_data(
        scaled, TRAINING_START_INDEX, TRADING_PERIOD_LENGTH, y_training_scaler)

    x_training_np = np.array(x_training)
    y_training_np = np.array(y_training)

    assert x_training_np.shape[0] == TRADING_PERIOD_LENGTH

    x_training_np_reshaped = np.reshape(
        x_training_np, (x_training_np.shape[0], 1, x_training_np.shape[1]))

    EPOCHS = 200
    BATCH_SIZE = 100

    # BATCH_SIZE must divide TRADING_PERIOD_LENGTH with no remainder
    assert TRADING_PERIOD_LENGTH % BATCH_SIZE == 0

    # TRADING_PERIOD_LENGTH must be less than half of the overall data (given even split between training and testing)
    assert TRADING_PERIOD_LENGTH <= 0.5 * scaled.shape[0]

    model = LstmAgent.get_model(BATCH_SIZE, x_training_np.shape[1])

    # loss_string = "mean_squared_logarithmic_error"
    loss_string = "mean_squared_error"
    # loss_string = "mean_absolute_error"
    # loss_string = "mean_absolute_percentage_error"
    model.compile(optimizer="adam", loss=loss_string)

    model.summary()

    x_testing, y_testing = TrainingDataHelper.get_x_y_data(
        scaled, scaled.shape[0] - TRADING_PERIOD_LENGTH, TRADING_PERIOD_LENGTH, MinMaxScaler())

    x_testing_np = np.array(x_testing)
    x_testing_np_reshaped = np.reshape(
        x_testing_np, (x_testing_np.shape[0], 1, x_testing_np.shape[1]))

    use_shuffle = False

    history = model.fit(x_training_np_reshaped, y_training_np,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_testing_np_reshaped, y_testing),
                        # validation_split=0.5
                        shuffle=use_shuffle,
                        verbose=2,
                        )

    y_predicted_on_testing_set = model.predict(
        x_testing_np_reshaped, batch_size=BATCH_SIZE)

    TrainingDataHelper.plot_results(TRAINING_START_INDEX, TRADING_PERIOD_LENGTH,
                                    BATCH_SIZE, EPOCHS,
                                    x_training_np_reshaped, y_training,
                                    y_testing, y_predicted_on_testing_set,
                                    model, loss_string, use_shuffle, history)


if __name__ == '__main__':
    main()
