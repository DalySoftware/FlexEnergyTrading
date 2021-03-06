from pandas.core.series import Series
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from NaturalGasDataProvider import NaturalGasDataProvider
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt  # type: ignore


class NeuralAgentHelper:
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

        return result

    @staticmethod
    def get_filtered_data(scaled, start_index, length):
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

        close_prices = x_copy["Close"]

        return x, y, close_prices

    @staticmethod
    def plot_results(TRAINING_START_INDEX, test_start_index,
                     TRADING_PERIOD_LENGTH, test_period_length,
                     BATCH_SIZE, EPOCHS, AGENT_TYPE,
                     IO_LAYER_UNITS, COMPRESSION_LAYER_UNITS,
                     x_training_np_reshaped, y_training,
                     y_validation, y_predicted_on_validation_set,
                     model, loss_string,  use_shuffle, fit_history,
                     mode, type):

        if type == "vali":
            type_str = "Validation"
        elif type == "test":
            type_str = "Testing"

        final_validation_loss = fit_history.history["val_loss"][-1]

        fig = plt.figure(figsize=[16, 9])
        x_values_for_training_plot = range(1, TRADING_PERIOD_LENGTH + 1)
        x_values_for_validation_plot = range(1, test_period_length + 1)

        training_plot = fig.add_subplot(2, 2, 1)
        validation_plot = fig.add_subplot(2, 2, 2)
        convergence_plot = fig.add_subplot(2, 2, 3)
        text_plot = fig.add_subplot(2, 2, 4)

        y_predicted_on_training_set = model.predict(
            x_training_np_reshaped, batch_size=BATCH_SIZE
        )
        # types: ignore
        training_plot.plot(x_values_for_training_plot, y_training,
                           x_values_for_training_plot, y_predicted_on_training_set)
        training_plot.set_xlabel("Day")
        training_plot.set_title("Training Data")
        training_plot.legend(["Actual", "Predicted"], loc="upper left")

        # types: ignore
        validation_plot.plot(x_values_for_validation_plot, y_validation,
                             x_values_for_validation_plot, y_predicted_on_validation_set)
        validation_plot.set_xlabel("Day")
        validation_plot.set_title(f"{type_str} Data")
        validation_plot.legend(["Actual", "Predicted"], loc="upper left")

        # types: ignore
        convergence_plot.plot(fit_history.history["loss"])
        convergence_plot.plot(fit_history.history["val_loss"])
        convergence_plot.set_title("Loss")
        convergence_plot.set_ylabel("Loss")
        convergence_plot.set_xlabel("Epoch")
        convergence_plot.legend(["Training", type_str], loc="upper left")

        if type == "vali":
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

        elif type == "test":
            info_text = f"\nAgent Type: {AGENT_TYPE}" + \
                        f"\nTesting Period Length: {test_period_length}" + \
                        f"\nTesting Start Index: {test_start_index}" + \
                        f"\nFinal {type_str} Loss: {round(final_validation_loss, 2)}"

        text_plot.text(0.5, 0.5, info_text,
                       horizontalalignment='left', verticalalignment='center',
                       )
        text_plot.set_title("Parameters")

        if type == "vali":
            filename = f"{AGENT_TYPE[:3]} {IO_LAYER_UNITS} {COMPRESSION_LAYER_UNITS} {IO_LAYER_UNITS} {EPOCHS} " + \
                       f"{TRAINING_START_INDEX} {TRADING_PERIOD_LENGTH} {BATCH_SIZE} {loss_string} {use_shuffle}"
        elif type == "test":
            filename = f"{AGENT_TYPE} {test_period_length} {test_start_index}"

        print(filename)

        if mode == "save":
            plt.savefig(
                f"C:\\Users\\irond\\Documents\\Coding\\FlexEnergyTrading\\other resources\\AgentOutput\\{type}\\{filename}.pdf",
                bbox_inches="tight")

        if mode == "show":
            plt.show()

    @staticmethod
    def filter_and_reshape_data(scaled_data, TRAINING_START_INDEX, TRADING_PERIOD_LENGTH):
        x_training, y_training, close_prices = NeuralAgentHelper.get_filtered_data(
            scaled_data, TRAINING_START_INDEX, TRADING_PERIOD_LENGTH)

        x_training_np = np.array(x_training)
        y_training_np = np.array(y_training)

        x_training_np_reshaped = np.reshape(
            x_training_np, (x_training_np.shape[0], 1, x_training_np.shape[1]))

        return y_training, x_training_np, y_training_np, x_training_np_reshaped, close_prices

    @staticmethod
    def get_vali_testing_start_indexes(trading_period_length, training_start_index):
        vali_start_index = training_start_index + trading_period_length
        testing_start_index = vali_start_index + trading_period_length
        return (vali_start_index, testing_start_index)

    @staticmethod
    def run_training(agent_type, trading_period_length, training_start_index, epochs, batch_size,
                     io_layer_units, compression_layer_units, loss_string, use_shuffle):
        data = NaturalGasDataProvider.get_data(True)

        scaled = NeuralAgentHelper.normalize_data(data)

        vali_start_index, _ = NeuralAgentHelper.get_vali_testing_start_indexes(
            trading_period_length, training_start_index)
        AVAILABLE_DATA_LENGTH = scaled.shape[0]

        print(f'AVAILABLE_DATA_LENGTH: {AVAILABLE_DATA_LENGTH}')

        y_training, x_training_np, y_training_np, x_training_np_reshaped, _ = NeuralAgentHelper.filter_and_reshape_data(
            scaled, training_start_index, trading_period_length
        )

        # batch_size must divide TRADING_PERIOD_LENGTH with no remainder
        assert trading_period_length % batch_size == 0

        # Ensure TRADING_PERIOD_LENGTH is less than one third of the overall data (for training, validation and cross-agent testing)
        assert trading_period_length <= 0.33 * AVAILABLE_DATA_LENGTH

        model = agent_type.get_model(
            batch_size, x_training_np.shape[1], io_layer_units, compression_layer_units)

        model.compile(optimizer="adam", loss=loss_string)

        model.summary()

        y_validation, _, _, x_validation_np_reshaped, _ = NeuralAgentHelper.filter_and_reshape_data(
            scaled, vali_start_index, trading_period_length
        )

        history = model.fit(x_training_np_reshaped, y_training_np,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(
                                x_validation_np_reshaped, y_validation),
                            shuffle=use_shuffle,
                            verbose=2,
                            )

        y_predicted_on_validation_set = model.predict(
            x_validation_np_reshaped, batch_size=batch_size)

        NeuralAgentHelper.plot_results(training_start_index, vali_start_index,
                                       trading_period_length, trading_period_length,
                                       batch_size, epochs,
                                       agent_type.__name__,
                                       io_layer_units, compression_layer_units,
                                       x_training_np_reshaped, y_training,
                                       y_validation, y_predicted_on_validation_set,
                                       model, loss_string, use_shuffle, history,
                                       'save', 'vali')

    @staticmethod
    def run_testing(agent_type, training_period_length, test_period_length, training_start_index, test_start_index, epochs, batch_size,
                    io_layer_units, compression_layer_units, loss_string, use_shuffle):
        data = NaturalGasDataProvider.get_data(True)

        scaled = NeuralAgentHelper.normalize_data(data)

        y_training, x_training_np, y_training_np, x_training_np_reshaped, _ = NeuralAgentHelper.filter_and_reshape_data(
            scaled, training_start_index, training_period_length
        )

        model = agent_type.get_model(
            batch_size, x_training_np.shape[1], io_layer_units, compression_layer_units)

        model.compile(optimizer="adam", loss=loss_string)

        model.summary()

        y_testing, _, _, x_testing_np_reshaped, test_close_prices = NeuralAgentHelper.filter_and_reshape_data(
            scaled, test_start_index, test_period_length
        )

        history = model.fit(x_training_np_reshaped, y_training_np,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(
                                x_testing_np_reshaped, y_testing),
                            shuffle=use_shuffle,
                            verbose=2,
                            )

        y_predicted_on_testing_set = model.predict(
            x_testing_np_reshaped, batch_size=batch_size)

        mode = ""
        if mode:
            NeuralAgentHelper.plot_results(training_start_index, test_start_index,
                                           training_period_length, test_period_length,
                                           batch_size, epochs,
                                           agent_type.__name__,
                                           io_layer_units, compression_layer_units,
                                           x_training_np_reshaped, y_training,
                                           y_testing, y_predicted_on_testing_set,
                                           model, loss_string, use_shuffle, history,
                                           mode, 'test')

        return y_predicted_on_testing_set, test_close_prices, history
