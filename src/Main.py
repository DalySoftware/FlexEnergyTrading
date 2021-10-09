from GruAgent import GruAgent
from LstmAgent import LstmAgent
from NeuralAgentHelper import NeuralAgentHelper

TRADING_PERIOD_LENGTH = 1800
TRAINING_START_INDEX = 300

run_training = NeuralAgentHelper.run_training
run_testing = NeuralAgentHelper.run_testing


def run_all_training():
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_error", False)

    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_error", False)

    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_logarithmic_error", False)

    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_logarithmic_error", False)

    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_error", True)

    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_error", True)

    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_logarithmic_error", True)

    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_logarithmic_error", True)

    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 20, 20, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 100, "mean_squared_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 100, "mean_squared_error", False)

    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 20, 20, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 100, "mean_squared_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 100, "mean_squared_error", False)

    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 100, "mean_squared_logarithmic_error", False)

    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 20, 20, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 100, "mean_squared_logarithmic_error", False)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 100, "mean_squared_logarithmic_error", False)

    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 20, 20, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 100, "mean_squared_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 100, "mean_squared_error", True)

    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 20, 20, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 100, "mean_squared_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 100, "mean_squared_error", True)

    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 100, "mean_squared_logarithmic_error", True)

    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 20, 20, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 100, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 200, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 300, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 600, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 900, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 75, 1800, 100, 100, "mean_squared_logarithmic_error", True)

    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 20, 300, 100, 100, "mean_squared_logarithmic_error", True)


def main():

    # loss_string = "mean_squared_logarithmic_error"
    # loss_string = "mean_squared_error"
    # loss_string = "mean_absolute_error"
    # loss_string = "mean_absolute_percentage_error"

    _, testing_overall_start_index = NeuralAgentHelper.get_vali_testing_start_indexes(
        TRADING_PERIOD_LENGTH, TRAINING_START_INDEX)

    # GRU Candidate 1
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 1800, TRAINING_START_INDEX, testing_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 900, TRAINING_START_INDEX, testing_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 900, TRAINING_START_INDEX, testing_overall_start_index + 900,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, testing_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, testing_overall_start_index + 600,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, testing_overall_start_index + 1200,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index + 300,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index + 600,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index + 900,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index + 1200,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(GruAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index + 1500,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)

    # LSTM Candidate 1
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 1800, TRAINING_START_INDEX, testing_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 900, TRAINING_START_INDEX, testing_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 900, TRAINING_START_INDEX, testing_overall_start_index + 900,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, testing_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, testing_overall_start_index + 600,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 600, TRAINING_START_INDEX, testing_overall_start_index + 1200,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index + 300,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index + 600,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index + 900,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)
    # run_testing(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, testing_overall_start_index + 1200,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)

    to_test = [
        # # GruAgent
        # [GruAgent, 1800, testing_overall_start_index],
        # [GruAgent,  900, testing_overall_start_index],
        # [GruAgent,  900, testing_overall_start_index + 900],
        # [GruAgent,  600, testing_overall_start_index],
        # [GruAgent,  600, testing_overall_start_index + 600],
        # [GruAgent,  600, testing_overall_start_index + 1200],
        # [GruAgent,  300, testing_overall_start_index],
        # [GruAgent,  300, testing_overall_start_index + 300],
        # [GruAgent,  300, testing_overall_start_index + 600],
        # [GruAgent,  300, testing_overall_start_index + 900],
        # [GruAgent,  300, testing_overall_start_index + 1200],
        # [GruAgent,  300, testing_overall_start_index + 1500],
        # # LstmAgent
        # [LstmAgent, 1800, testing_overall_start_index],
        # [LstmAgent,  900, testing_overall_start_index],
        # [LstmAgent,  900, testing_overall_start_index + 900],
        # [LstmAgent,  600, testing_overall_start_index],
        # [LstmAgent,  600, testing_overall_start_index + 600],
        # [LstmAgent,  600, testing_overall_start_index + 1200],
        # [LstmAgent,  300, testing_overall_start_index],
        # [LstmAgent,  300, testing_overall_start_index + 300],
        # [LstmAgent,  300, testing_overall_start_index + 600],
        # [LstmAgent,  300, testing_overall_start_index + 900],
        # [LstmAgent,  300, testing_overall_start_index + 1200],
        # [LstmAgent,  300, testing_overall_start_index + 1500],
        # PerfectPredictionAgent
        # [PerfectPredictionAgent, 1800, testing_overall_start_index],
        # [PerfectPredictionAgent,  900, testing_overall_start_index],
        # [PerfectPredictionAgent,  900, testing_overall_start_index + 900],
        # [PerfectPredictionAgent,  600, testing_overall_start_index],
        # [PerfectPredictionAgent,  600, testing_overall_start_index + 600],
        # [PerfectPredictionAgent,  600, testing_overall_start_index + 1200],
        # [PerfectPredictionAgent,  300, testing_overall_start_index],
        # [PerfectPredictionAgent,  300, testing_overall_start_index + 300],
        # [PerfectPredictionAgent,  300, testing_overall_start_index + 600],
        # [PerfectPredictionAgent,  300, testing_overall_start_index + 900],
        # [PerfectPredictionAgent,  300, testing_overall_start_index + 1200],
        # [PerfectPredictionAgent,  300, testing_overall_start_index + 1500],
        # # PerfectKnowledgeAgent
        # [PerfectKnowledgeAgent, 1800, testing_overall_start_index],
        # [PerfectKnowledgeAgent,  900, testing_overall_start_index],
        # [PerfectKnowledgeAgent,  900, testing_overall_start_index + 900],
        # [PerfectKnowledgeAgent,  600, testing_overall_start_index],
        # [PerfectKnowledgeAgent,  600, testing_overall_start_index + 600],
        # [PerfectKnowledgeAgent,  600, testing_overall_start_index + 1200],
        # [PerfectKnowledgeAgent,  300, testing_overall_start_index],
        # [PerfectKnowledgeAgent,  300, testing_overall_start_index + 300],
        # [PerfectKnowledgeAgent,  300, testing_overall_start_index + 600],
        # [PerfectKnowledgeAgent,  300, testing_overall_start_index + 900],
        # [PerfectKnowledgeAgent,  300, testing_overall_start_index + 1200],
        # [PerfectKnowledgeAgent,  300, testing_overall_start_index + 1500],
        # DayZeroAgent
        # [DayZeroAgent, 1800, testing_overall_start_index],
        # [DayZeroAgent,  900, testing_overall_start_index],
        # [DayZeroAgent,  900, testing_overall_start_index + 900],
        # [DayZeroAgent,  600, testing_overall_start_index],
        # [DayZeroAgent,  600, testing_overall_start_index + 600],
        # [DayZeroAgent,  600, testing_overall_start_index + 1200],
        # [DayZeroAgent,  300, testing_overall_start_index],
        # [DayZeroAgent,  300, testing_overall_start_index + 300],
        # [DayZeroAgent,  300, testing_overall_start_index + 600],
        # [DayZeroAgent,  300, testing_overall_start_index + 900],
        # [DayZeroAgent,  300, testing_overall_start_index + 1200],
        # [DayZeroAgent,  300, testing_overall_start_index + 1500],
        # EqualIntervalAgent
        # [EqualIntervalAgent, 1800, testing_overall_start_index],
        # [EqualIntervalAgent,  900, testing_overall_start_index],
        # [EqualIntervalAgent,  900, testing_overall_start_index + 900],
        # [EqualIntervalAgent,  600, testing_overall_start_index],
        # [EqualIntervalAgent,  600, testing_overall_start_index + 600],
        # [EqualIntervalAgent,  600, testing_overall_start_index + 1200],
        # [EqualIntervalAgent,  300, testing_overall_start_index],
        # [EqualIntervalAgent,  300, testing_overall_start_index + 300],
        # [EqualIntervalAgent,  300, testing_overall_start_index + 600],
        # [EqualIntervalAgent,  300, testing_overall_start_index + 900],
        # [EqualIntervalAgent,  300, testing_overall_start_index + 1200],
        # [EqualIntervalAgent,  300, testing_overall_start_index + 1500],

    ]

    to_print = []

    for testing_params in to_test:
        agent_type = testing_params[0]
        period_length = testing_params[1]
        period_start = testing_params[2]

        hedged_volume, total_cost = agent_type.get_trade_performance(
            period_length, period_start)

        to_print.append(
            f"agent: {agent_type.__name__}, period length: {period_length}, period start: {period_start}, volume: {hedged_volume}, cost: {total_cost}")

    for line in to_print:
        print(line)

    print('done')


if __name__ == '__main__':
    # main()
    run_all_training()
