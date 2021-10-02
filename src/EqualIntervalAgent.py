from NaturalGasDataProvider import NaturalGasDataProvider
from NeuralAgentHelper import NeuralAgentHelper
from TradeSignalHelper import TOTAL_VOLUME, TradeSignalHelper

TOTAL_VOLUME = 1
NUMBER_TRADES = 10


class EqualIntervalAgent:
    @staticmethod
    def get_trade_performance(eval_period_length, evaluation_start_index):
        data = NaturalGasDataProvider.get_data(True)
        scaled = NeuralAgentHelper.normalize_data(data)

        y_evaluation, _, _, x_evaluation_np_reshaped, eval_close_prices = NeuralAgentHelper.filter_and_reshape_data(
            scaled, evaluation_start_index, eval_period_length
        )

        interval = eval_period_length // (NUMBER_TRADES - 1)

        trade_volumes = [0] * eval_period_length

        trades_made = 0
        current_index = 0
        while trades_made < NUMBER_TRADES - 1:
            trade_volumes[current_index] = TOTAL_VOLUME / NUMBER_TRADES
            trades_made += 1
            current_index += interval

        print(trades_made)

        hedged_volume, total_cost = TradeSignalHelper.get_trade_performance(
            eval_close_prices, trade_volumes)

        return hedged_volume, total_cost
