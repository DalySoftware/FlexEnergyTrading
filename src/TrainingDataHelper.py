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
    def plot_results(TRAINING_START_INDEX, validation_start_index,
                     TRADING_PERIOD_LENGTH, validation_period_length,
                     BATCH_SIZE, EPOCHS, AGENT_TYPE,
                     IO_LAYER_UNITS, COMPRESSION_LAYER_UNITS,
                     x_training_np_reshaped, y_training,
                     y_testing, y_predicted_on_testing_set,
                     model, loss_string,  use_shuffle, fit_history,
                     mode, type):

        if type == "testing":
            type_str = "Testing"
        elif type == "eval":
            type_str = "Evaluation"

        final_validation_loss = fit_history.history["val_loss"][-1]

        fig = plt.figure(figsize=[16, 9])
        x_values_for_training_plot = range(1, TRADING_PERIOD_LENGTH + 1)
        x_values_for_test_plot = range(1, validation_period_length + 1)

        training_plot = fig.add_subplot(2, 2, 1)
        testing_plot = fig.add_subplot(2, 2, 2)
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
        testing_plot.plot(x_values_for_test_plot, y_testing,
                          x_values_for_test_plot, y_predicted_on_testing_set)
        testing_plot.set_xlabel("Day")
        testing_plot.set_title(f"{type_str} Data")
        testing_plot.legend(["Actual", "Predicted"], loc="upper left")

        # types: ignore
        convergence_plot.plot(fit_history.history["loss"])
        convergence_plot.plot(fit_history.history["val_loss"])
        convergence_plot.set_title("Loss")
        convergence_plot.set_ylabel("Loss")
        convergence_plot.set_xlabel("Epoch")
        convergence_plot.legend(["Training", type_str], loc="upper left")

        if type == "testing":
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

        elif type == "eval":
            info_text = f"\nAgent Type: {AGENT_TYPE}" + \
                        f"\nEvaluation Period Length: {validation_period_length}" + \
                        f"\nEvaluation Start Index: {validation_start_index}" + \
                        f"\nFinal {type_str} Loss: {round(final_validation_loss, 2)}"

        text_plot.text(0.5, 0.5, info_text,
                       horizontalalignment='left', verticalalignment='center',
                       )
        text_plot.set_title("Parameters")

        if type == "testing":
            filename = f"{AGENT_TYPE[:3]} {IO_LAYER_UNITS} {COMPRESSION_LAYER_UNITS} {IO_LAYER_UNITS} {EPOCHS} " + \
                       f"{TRAINING_START_INDEX} {TRADING_PERIOD_LENGTH} {BATCH_SIZE} {loss_string} {use_shuffle}"
        elif type == "eval":
            filename = f"{AGENT_TYPE} {validation_period_length} {validation_start_index}"

        print(filename)

        if mode == "save":
            plt.savefig(
                f"C:\\Users\\irond\\Documents\\Coding\\FlexEnergyTrading\\other resources\\AgentOutput\\{type}\\{filename}.pdf",
                bbox_inches="tight")

        if mode == "show":
            plt.show()

    @staticmethod
    def get_and_reshape_data(scaled, TRAINING_START_INDEX, TRADING_PERIOD_LENGTH):
        x_training, y_training = TrainingDataHelper.get_x_y_data(
            scaled, TRAINING_START_INDEX, TRADING_PERIOD_LENGTH)

        x_training_np = np.array(x_training)
        y_training_np = np.array(y_training)

        x_training_np_reshaped = np.reshape(
            x_training_np, (x_training_np.shape[0], 1, x_training_np.shape[1]))

        return y_training, x_training_np, y_training_np, x_training_np_reshaped

    @staticmethod
    def get_testing_eval_start_indexes(trading_period_length, training_start_index):
        testing_start_index = training_start_index + trading_period_length
        evaluation_start_index = testing_start_index + trading_period_length
        return (testing_start_index, evaluation_start_index)

    @staticmethod
    def run_training(agent_type, trading_period_length, training_start_index, epochs, batch_size,
                     io_layer_units, compression_layer_units, loss_string, use_shuffle):
        data = NaturalGasDataProvider.get_data(True)

        scaled = TrainingDataHelper.normalize_data(data)

        testing_start_index, _ = TrainingDataHelper.get_testing_eval_start_indexes(
            trading_period_length, training_start_index)
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

        y_testing, _, _, x_testing_np_reshaped = TrainingDataHelper.get_and_reshape_data(
            scaled, testing_start_index, trading_period_length
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

        TrainingDataHelper.plot_results(training_start_index, testing_start_index,
                                        trading_period_length, trading_period_length,
                                        batch_size, epochs,
                                        agent_type.__name__,
                                        io_layer_units, compression_layer_units,
                                        x_training_np_reshaped, y_training,
                                        y_testing, y_predicted_on_testing_set,
                                        model, loss_string, use_shuffle, history,
                                        'save', 'testing')

    @staticmethod
    def run_evaluation(agent_type, training_period_length, eval_period_length, training_start_index, evaluation_start_index, epochs, batch_size,
                       io_layer_units, compression_layer_units, loss_string, use_shuffle):
        data = NaturalGasDataProvider.get_data(True)

        scaled = TrainingDataHelper.normalize_data(data)

        y_training, x_training_np, y_training_np, x_training_np_reshaped = TrainingDataHelper.get_and_reshape_data(
            scaled, training_start_index, training_period_length
        )

        model = agent_type.get_model(
            batch_size, x_training_np.shape[1], io_layer_units, compression_layer_units)

        model.compile(optimizer="adam", loss=loss_string)

        model.summary()

        y_evaluation, _, _, x_evaluation_np_reshaped = TrainingDataHelper.get_and_reshape_data(
            scaled, evaluation_start_index, eval_period_length
        )

        history = model.fit(x_training_np_reshaped, y_training_np,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(
                                x_evaluation_np_reshaped, y_evaluation),
                            shuffle=use_shuffle,
                            verbose=2,
                            )

        y_predicted_on_evaluation_set = model.predict(
            x_evaluation_np_reshaped, batch_size=batch_size)

        TrainingDataHelper.plot_results(training_start_index, evaluation_start_index,
                                        training_period_length, eval_period_length,
                                        batch_size, epochs,
                                        agent_type.__name__,
                                        io_layer_units, compression_layer_units,
                                        x_training_np_reshaped, y_training,
                                        y_evaluation, y_predicted_on_evaluation_set,
                                        model, loss_string, use_shuffle, history,
                                        'save', 'eval')

        return y_predicted_on_evaluation_set, history
