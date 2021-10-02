from NaturalGasDataProvider import NaturalGasDataProvider
from NeuralAgentHelper import NeuralAgentHelper
from TradeSignalHelper import TradeSignalHelper


class PerfectPredictionAgent:
    @staticmethod
    def run_evaluation(agent_type, training_period_length, eval_period_length, training_start_index, evaluation_start_index, epochs, batch_size,
                       io_layer_units, compression_layer_units, loss_string, use_shuffle):
        data = NaturalGasDataProvider.get_data(True)
        scaled = NeuralAgentHelper.normalize_data(data)

        y_evaluation, _, _, x_evaluation_np_reshaped, eval_close_prices = NeuralAgentHelper.filter_and_reshape_data(
            scaled, evaluation_start_index, eval_period_length
        )

        y_values = y_evaluation["DiffFromGlobalMinScaled"]

        # this matches the output of the neural net solutions
        y_values_formatted = [[y] for y in y_values]

        return y_values_formatted, eval_close_prices, None

    @staticmethod
    def get_trade_performance(eval_period_length, evaluation_start_index):
        y_predicted, close_prices, history = PerfectPredictionAgent.run_evaluation(PerfectPredictionAgent, None, eval_period_length, None, evaluation_start_index,
                                                                                   None, None, None, None, None, None)

        y_predicted_vector = [y[0] for y in y_predicted]

        trade_volumes = TradeSignalHelper.get_trade_amounts_for_sigmoid(
            y_predicted_vector)

        hedged_volume, total_cost = TradeSignalHelper.get_trade_performance(
            close_prices, trade_volumes)

        return hedged_volume, total_cost
