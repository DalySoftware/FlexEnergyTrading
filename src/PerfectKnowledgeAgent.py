from NaturalGasDataProvider import NaturalGasDataProvider
from NeuralAgentHelper import NeuralAgentHelper

TOTAL_VOLUME = 1


class PerfectKnowledgeAgent:
    @staticmethod
    def get_trade_performance(test_period_length, test_start_index):
        data = NaturalGasDataProvider.get_data(True)
        scaled = NeuralAgentHelper.normalize_data(data)

        y_testing, _, _, x_testing_np_reshaped, test_close_prices = NeuralAgentHelper.filter_and_reshape_data(
            scaled, test_start_index, test_period_length
        )

        min_price = test_close_prices.min()

        hedged_volume = TOTAL_VOLUME
        total_cost = hedged_volume * min_price

        return hedged_volume, total_cost
