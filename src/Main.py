from GruAgent import GruAgent
from LstmAgent import LstmAgent
from TrainingDataHelper import TrainingDataHelper


def main():
    TRADING_PERIOD_LENGTH = 1800
    TRAINING_START_INDEX = 300

    # loss_string = "mean_squared_logarithmic_error"
    # loss_string = "mean_squared_error"
    # loss_string = "mean_absolute_error"
    # loss_string = "mean_absolute_percentage_error"

    run_training = TrainingDataHelper.run_training
    run_evaluation = TrainingDataHelper.run_evaluation

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

    _, evaluation_overall_start_index = TrainingDataHelper.get_testing_eval_start_indexes(
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
    run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 1800, TRAINING_START_INDEX, evaluation_overall_start_index,
                   20, 300, 100, 100, "mean_squared_logarithmic_error", True)
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
    # run_evaluation(LstmAgent, TRADING_PERIOD_LENGTH, 300, TRAINING_START_INDEX, evaluation_overall_start_index + 1500,
    #                20, 300, 100, 100, "mean_squared_logarithmic_error", True)


if __name__ == '__main__':
    main()
