from NaturalGasDataProvider import NaturalGasDataProvider
from NeuralAgentHelper import NeuralAgentHelper

TOTAL_VOLUME = 1


class DayZeroAgent:
    @staticmethod
    def get_trade_performance(eval_period_length, evaluation_start_index):
        data = NaturalGasDataProvider.get_data(True)
        scaled = NeuralAgentHelper.normalize_data(data)

        y_evaluation, _, _, x_evaluation_np_reshaped, eval_close_prices = NeuralAgentHelper.filter_and_reshape_data(
            scaled, evaluation_start_index, eval_period_length
        )

        trade_price = eval_close_prices[0]

        hedged_volume = TOTAL_VOLUME
        total_cost = hedged_volume * trade_price

        return hedged_volume, total_cost
