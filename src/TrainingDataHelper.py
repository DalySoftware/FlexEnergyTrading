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

        if (np.any(np.isnan(result[column]))):  # type: ignore
            result[column] = np.zeros(result[column].shape)  # type: ignore

        if (not np.all(np.isfinite(result[column]))):   # type: ignore
            result[column] = np.zeros(result[column].shape)  # type: ignore

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
    def get_x_y_data(scaled, start_index, length):
        scaler = MinMaxScaler()

        x: DataFrame = scaled[start_index:start_index + length].copy()

        x["DaysLeftInWindow"] = np.flip(np.arange(length))
        x["RunningMinClose"] = x["Close"].expanding().min()

        x_copy = x.copy()
        x_copy["GlobalMin"] = x_copy["Close"].min()

        x_copy["DiffFromGlobalMin"] = x_copy.loc[:, ["Close", "GlobalMin"]].apply(
            lambda row: row["Close"] - row["GlobalMin"], axis=1)

        x_copy["DiffFromGlobalMinScaled"] = scaler.fit_transform(
            x_copy["DiffFromGlobalMin"].values.reshape(-1, 1))

        y: DataFrame = x_copy.loc[:, ["DiffFromGlobalMinScaled"]]

        return x, y

    @staticmethod
    def plot_results(TRAINING_START_INDEX, TRADING_PERIOD_LENGTH,
                     BATCH_SIZE, EPOCHS, AGENT_TYPE,
                     IO_LAYER_UNITS, COMPRESSION_LAYER_UNITS,
                     x_training_np_reshaped, y_training,
                     y_testing, y_predicted_on_testing_set,
                     model, loss_string,  use_shuffle, fit_history):

        fig = plt.figure(figsize=[16, 9])
        x_values_for_plot = range(1, TRADING_PERIOD_LENGTH + 1)

        training_plot = fig.add_subplot(2, 2, 1)
        testing_plot = fig.add_subplot(2, 2, 2)
        convergence_plot = fig.add_subplot(2, 2, 3)
        text_plot = fig.add_subplot(2, 2, 4)

        y_predicted_on_training_set = model.predict(
            x_training_np_reshaped, batch_size=BATCH_SIZE
        )
        # types: ignore
        training_plot.plot(x_values_for_plot, y_training,
                           x_values_for_plot, y_predicted_on_training_set)
        training_plot.set_xlabel("Day")
        training_plot.set_title("Training Data")
        training_plot.legend(["Training", "Predicted"], loc="upper left")

        # types: ignore
        testing_plot.plot(x_values_for_plot, y_testing,
                          x_values_for_plot, y_predicted_on_testing_set)
        testing_plot.set_xlabel("Day")
        testing_plot.set_title("Testing Data")
        training_plot.legend(["Testing", "Predicted"], loc="upper left")

        # types: ignore
        convergence_plot.plot(fit_history.history["loss"])
        convergence_plot.plot(fit_history.history["val_loss"])
        convergence_plot.set_title("Loss")
        convergence_plot.set_ylabel("Loss")
        convergence_plot.set_xlabel("Epoch")
        convergence_plot.legend(["Train", "Test"], loc="upper left")

        info_text = f"\nAgent Type: {AGENT_TYPE}" + \
                    f"\nUnits Layer 1: {IO_LAYER_UNITS}" + \
                    f"\nUnits Layer 2: {COMPRESSION_LAYER_UNITS}" + \
                    f"\nUnits Layer 3: {IO_LAYER_UNITS}" + \
                    f"\nEpochs: {EPOCHS}  " + \
                    f"\nTraining Start Index: {TRAINING_START_INDEX}  " + \
                    f"\nTrading Period Length: {TRADING_PERIOD_LENGTH}" + \
                    f"\nBatch Size: {BATCH_SIZE}" + \
                    f"\nLoss Function: {loss_string}" + \
                    f"\nShuffle: {use_shuffle}"

        text_plot.text(0.5, 0.5, info_text,
                       horizontalalignment='left', verticalalignment='center',
                       )
        text_plot.set_title("Parameters")

        filename = f"{AGENT_TYPE[:3]} {IO_LAYER_UNITS} {COMPRESSION_LAYER_UNITS} {IO_LAYER_UNITS} {EPOCHS} " + \
                   f"{TRAINING_START_INDEX} {TRADING_PERIOD_LENGTH} {BATCH_SIZE} {loss_string} {use_shuffle}"

        plt.savefig(
            f"C:\\Users\\irond\\Documents\\Coding\\FlexEnergyTrading\\other resources\\AgentOutput\\{filename}.pdf",
            bbox_inches="tight")
        # plt.show()

    @staticmethod
    def get_and_reshape_data(scaled, TRAINING_START_INDEX, TRADING_PERIOD_LENGTH):
        x_training, y_training = TrainingDataHelper.get_x_y_data(
            scaled, TRAINING_START_INDEX, TRADING_PERIOD_LENGTH)

        x_training_np = np.array(x_training)
        y_training_np = np.array(y_training)

        x_training_np_reshaped = np.reshape(
            x_training_np, (x_training_np.shape[0], 1, x_training_np.shape[1]))

        return y_training, x_training_np, y_training_np, x_training_np_reshaped


def run_training(agent_type, trading_period_length, training_start_index, epochs, batch_size,
                 io_layer_units, compression_layer_units, loss_string, use_shuffle):
    data = NaturalGasDataProvider.get_data(True)

    scaled = TrainingDataHelper.normalize_data(data)

    # print(scaled[200:].head())
    # print(scaled[200:].tail())
    # scaled.to_csv("C:\\temp\\temp.csv")

    TESTING_START_INDEX = training_start_index + trading_period_length
    EVALUATION_START_INDEX = TESTING_START_INDEX + trading_period_length
    AVAILABLE_DATA_LENGTH = scaled.shape[0]

    print(f'AVAILABLE_DATA_LENGTH: {AVAILABLE_DATA_LENGTH}')

    y_training, x_training_np, y_training_np, x_training_np_reshaped = TrainingDataHelper.get_and_reshape_data(
        scaled, training_start_index, trading_period_length
    )

    # batch_size must divide TRADING_PERIOD_LENGTH with no remainder
    assert trading_period_length % batch_size == 0

    # Ensure TRADING_PERIOD_LENGTH is less than one third of the overall data (for training, testing and cross-agent evaluation)
    assert trading_period_length <= 0.33 * AVAILABLE_DATA_LENGTH

    model = agent_type.get_model(
        batch_size, x_training_np.shape[1], io_layer_units, compression_layer_units)

    model.compile(optimizer="adam", loss=loss_string)

    model.summary()

    y_testing, x_testing_np, y_testing_np, x_testing_np_reshaped = TrainingDataHelper.get_and_reshape_data(
        scaled, TESTING_START_INDEX, trading_period_length
    )

    history = model.fit(x_training_np_reshaped, y_training_np,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_testing_np_reshaped, y_testing),
                        shuffle=use_shuffle,
                        verbose=2,
                        )

    y_predicted_on_testing_set = model.predict(
        x_testing_np_reshaped, batch_size=batch_size)

    TrainingDataHelper.plot_results(training_start_index, trading_period_length,
                                    batch_size, epochs,
                                    agent_type.__name__,
                                    io_layer_units, compression_layer_units,
                                    x_training_np_reshaped, y_training,
                                    y_testing, y_predicted_on_testing_set,
                                    model, loss_string, use_shuffle, history)
