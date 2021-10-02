from numpy import e

TOTAL_VOLUME = 1


class TradeSignalHelper:
    @staticmethod
    def get_trade_amount_from_sigmoid(difference_from_minimum):
        """Return the amount to trade given a scalar 'difference from minimum' input value (actual or predicted)."""

        x = difference_from_minimum

        a = -7
        b = 20
        k = 1

        y = k / (1 + e ** (a + b * x))

        return y

    @staticmethod
    def get_trade_amounts_for_sigmoid(difference_from_minimum_vector):
        """Return the amount to trade given a vector of 'difference from minimum' input values (actual or predicted)."""

        trade_volumes = [TradeSignalHelper.get_trade_amount_from_sigmoid(
            x) for x in difference_from_minimum_vector]

        return trade_volumes

    @staticmethod
    def unhedged_volume(hedged_volume):
        return TOTAL_VOLUME - hedged_volume

    @staticmethod
    def action_trade(hedged_volume, total_cost, trade_price, trade_volume):
        unhedged_volume = TradeSignalHelper.unhedged_volume(hedged_volume)

        if 0 < unhedged_volume <= trade_volume:
            trade_volume = unhedged_volume

        new_hedged_volume = hedged_volume + trade_volume
        new_total_cost = total_cost + trade_volume * trade_price

        return new_hedged_volume, new_total_cost

    @staticmethod
    def get_trade_performance(close_prices, trade_volumes):
        print("Getting trade performance")

        hedged_volume = 0
        total_cost = 0

        i = 0

        while i < len(close_prices) and hedged_volume < TOTAL_VOLUME:
            if i % 100 == 0:
                print(f"Record {i}")

            trade_price = close_prices[i]
            trade_volume = trade_volumes[i]

            hedged_volume, total_cost = TradeSignalHelper.action_trade(hedged_volume, total_cost,
                                                                       trade_price, trade_volume)

            # print(hedged_volume, total_cost)

            i += 1

        print(f"hedged volume: {hedged_volume}, total cost: {total_cost}")

        if hedged_volume < 1:
            # trade all remaining volume at the last price
            final_trade_price = close_prices[-1]
            final_volume = TradeSignalHelper.unhedged_volume(hedged_volume)

            hedged_volume, total_cost = TradeSignalHelper.action_trade(
                hedged_volume, total_cost, final_trade_price, final_volume)

        print(f"hedged volume: {hedged_volume}, total cost: {total_cost}")

        return hedged_volume, total_cost
