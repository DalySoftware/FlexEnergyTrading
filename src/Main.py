from DayZeroAgent import DayZeroAgent
from EqualIntervalAgent import EqualIntervalAgent
from GruAgent import GruAgent
from LstmAgent import LstmAgent
from PerfectKnowledgeAgent import PerfectKnowledgeAgent
from PerfectPredictionAgent import PerfectPredictionAgent
from TradeSignalHelper import TradeSignalHelper
from NeuralAgentHelper import NeuralAgentHelper


def main():
    TRADING_PERIOD_LENGTH = 1800
    TRAINING_START_INDEX = 300

    # loss_string = "mean_squared_logarithmic_error"
    # loss_string = "mean_squared_error"
    # loss_string = "mean_absolute_error"
    # loss_string = "mean_absolute_percentage_error"

    run_training = NeuralAgentHelper.run_training
    run_evaluation = NeuralAgentHelper.run_evaluation

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_error", False)

    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_error", False)

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_logarithmic_error", False)

    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_logarithmic_error", False)

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_error", True)

    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_error", True)

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_logarithmic_error", True)

    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_logarithmic_error", True)

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 20, 20, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 100, "mean_squared_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 100, "mean_squared_error", False)

    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 20, 20, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 100, "mean_squared_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 100, "mean_squared_error", False)

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 100, "mean_squared_logarithmic_error", False)

    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 20, 20, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 100, "mean_squared_logarithmic_error", False)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 100, "mean_squared_logarithmic_error", False)

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 20, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 100, "mean_squared_error", True)

    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 20, 20, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 100, "mean_squared_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 100, "mean_squared_error", True)

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 100, "mean_squared_logarithmic_error", True)

    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 20, 20, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 100, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 200, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 600, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 900, 100, 100, "mean_squared_logarithmic_error", True)
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              75, 1800, 100, 100, "mean_squared_logarithmic_error", True)

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              20, 300, 100, 100, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              20, 300, 100, 20, "mean_squared_error", True)
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              20, 300, 100, 20, "mean_squared_logarithmic_error", True)

    _, evaluation_overall_start_index = NeuralAgentHelper.get_testing_eval_start_indexes(
        TRADING_PERIOD_LENGTH, TRAINING_START_INDEX)

    # GRU Candidate 1
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 1800, TRAINING_START_INDEX, evaluation_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 900, TRAINING_START_INDEX, evaluation_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 900, TRAINING_START_INDEX, evaluation_overall_start_index + 900,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, evaluation_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, evaluation_overall_start_index + 600,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, evaluation_overall_start_index + 1200,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 300,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 600,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 900,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 1200,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 1500,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)

    # LSTM Candidate 1
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 1800, TRAINING_START_INDEX, evaluation_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 900, TRAINING_START_INDEX, evaluation_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 900, TRAINING_START_INDEX, evaluation_overall_start_index + 900,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, evaluation_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, evaluation_overall_start_index + 600,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, evaluation_overall_start_index + 1200,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 300,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 600,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 900,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 1200,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)

    to_evaluate = [
        # # GruAgent
        # [GruAgent, 1800, evaluation_overall_start_index],
        # [GruAgent,  900, evaluation_overall_start_index],
        # [GruAgent,  900, evaluation_overall_start_index + 900],
        # [GruAgent,  600, evaluation_overall_start_index],
        # [GruAgent,  600, evaluation_overall_start_index + 600],
        # [GruAgent,  600, evaluation_overall_start_index + 1200],
        # [GruAgent,  300, evaluation_overall_start_index],
        # [GruAgent,  300, evaluation_overall_start_index + 300],
        # [GruAgent,  300, evaluation_overall_start_index + 600],
        # [GruAgent,  300, evaluation_overall_start_index + 900],
        # [GruAgent,  300, evaluation_overall_start_index + 1200],
        # [GruAgent,  300, evaluation_overall_start_index + 1500],
        # # LstmAgent
        # [LstmAgent, 1800, evaluation_overall_start_index],
        # [LstmAgent,  900, evaluation_overall_start_index],
        # [LstmAgent,  900, evaluation_overall_start_index + 900],
        # [LstmAgent,  600, evaluation_overall_start_index],
        # [LstmAgent,  600, evaluation_overall_start_index + 600],
        # [LstmAgent,  600, evaluation_overall_start_index + 1200],
        # [LstmAgent,  300, evaluation_overall_start_index],
        # [LstmAgent,  300, evaluation_overall_start_index + 300],
        # [LstmAgent,  300, evaluation_overall_start_index + 600],
        # [LstmAgent,  300, evaluation_overall_start_index + 900],
        # [LstmAgent,  300, evaluation_overall_start_index + 1200],
        # [LstmAgent,  300, evaluation_overall_start_index + 1500],
        # PerfectPredictionAgent
        # [PerfectPredictionAgent, 1800, evaluation_overall_start_index],
        # [PerfectPredictionAgent,  900, evaluation_overall_start_index],
        # [PerfectPredictionAgent,  900, evaluation_overall_start_index + 900],
        # [PerfectPredictionAgent,  600, evaluation_overall_start_index],
        # [PerfectPredictionAgent,  600, evaluation_overall_start_index + 600],
        # [PerfectPredictionAgent,  600, evaluation_overall_start_index + 1200],
        # [PerfectPredictionAgent,  300, evaluation_overall_start_index],
        # [PerfectPredictionAgent,  300, evaluation_overall_start_index + 300],
        # [PerfectPredictionAgent,  300, evaluation_overall_start_index + 600],
        # [PerfectPredictionAgent,  300, evaluation_overall_start_index + 900],
        # [PerfectPredictionAgent,  300, evaluation_overall_start_index + 1200],
        # [PerfectPredictionAgent,  300, evaluation_overall_start_index + 1500],
        # # PerfectKnowledgeAgent
        # [PerfectKnowledgeAgent, 1800, evaluation_overall_start_index],
        # [PerfectKnowledgeAgent,  900, evaluation_overall_start_index],
        # [PerfectKnowledgeAgent,  900, evaluation_overall_start_index + 900],
        # [PerfectKnowledgeAgent,  600, evaluation_overall_start_index],
        # [PerfectKnowledgeAgent,  600, evaluation_overall_start_index + 600],
        # [PerfectKnowledgeAgent,  600, evaluation_overall_start_index + 1200],
        # [PerfectKnowledgeAgent,  300, evaluation_overall_start_index],
        # [PerfectKnowledgeAgent,  300, evaluation_overall_start_index + 300],
        # [PerfectKnowledgeAgent,  300, evaluation_overall_start_index + 600],
        # [PerfectKnowledgeAgent,  300, evaluation_overall_start_index + 900],
        # [PerfectKnowledgeAgent,  300, evaluation_overall_start_index + 1200],
        # [PerfectKnowledgeAgent,  300, evaluation_overall_start_index + 1500],
        # DayZeroAgent
        # [DayZeroAgent, 1800, evaluation_overall_start_index],
        [DayZeroAgent,  900, evaluation_overall_start_index],
        # [DayZeroAgent,  900, evaluation_overall_start_index + 900],
        # [DayZeroAgent,  600, evaluation_overall_start_index],
        # [DayZeroAgent,  600, evaluation_overall_start_index + 600],
        # [DayZeroAgent,  600, evaluation_overall_start_index + 1200],
        # [DayZeroAgent,  300, evaluation_overall_start_index],
        # [DayZeroAgent,  300, evaluation_overall_start_index + 300],
        # [DayZeroAgent,  300, evaluation_overall_start_index + 600],
        # [DayZeroAgent,  300, evaluation_overall_start_index + 900],
        # [DayZeroAgent,  300, evaluation_overall_start_index + 1200],
        # [DayZeroAgent,  300, evaluation_overall_start_index + 1500],
        # EqualIntervalAgent
        # [EqualIntervalAgent, 1800, evaluation_overall_start_index],
        [EqualIntervalAgent,  900, evaluation_overall_start_index],
        # [EqualIntervalAgent,  900, evaluation_overall_start_index + 900],
        # [EqualIntervalAgent,  600, evaluation_overall_start_index],
        # [EqualIntervalAgent,  600, evaluation_overall_start_index + 600],
        # [EqualIntervalAgent,  600, evaluation_overall_start_index + 1200],
        # [EqualIntervalAgent,  300, evaluation_overall_start_index],
        # [EqualIntervalAgent,  300, evaluation_overall_start_index + 300],
        # [EqualIntervalAgent,  300, evaluation_overall_start_index + 600],
        # [EqualIntervalAgent,  300, evaluation_overall_start_index + 900],
        # [EqualIntervalAgent,  300, evaluation_overall_start_index + 1200],
        # [EqualIntervalAgent,  300, evaluation_overall_start_index + 1500],

    ]

    to_print = []

    for evaluation_params in to_evaluate:
        agent_type = evaluation_params[0]
        period_length = evaluation_params[1]
        period_start = evaluation_params[2]

        hedged_volume, total_cost = agent_type.get_trade_performance(
            period_length, period_start)

        to_print.append(
            f"agent: {agent_type.__name__}, period length: {period_length}, period start: {period_start}, volume: {hedged_volume}, cost: {total_cost}")

    for line in to_print:
        print(line)

    print('done')


if __name__ == '__main__':
    main()
