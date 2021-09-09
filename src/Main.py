from GruAgent import GruAgent
from LstmAgent import LstmAgent
from TrainingDataHelper import run_training


def main():
    TRADING_PERIOD_LENGTH = 1800
    TRAINING_START_INDEX = 300

    # loss_string = "mean_squared_logarithmic_error"
    # loss_string = "mean_squared_error"
    # loss_string = "mean_absolute_error"
    # loss_string = "mean_absolute_percentage_error"

    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_error")
    # run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_error")

    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 20, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 20, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 20, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 20, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 20, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 20, 20, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 100, 100, 100, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 200, 100, 100, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 300, 100, 100, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 600, 100, 100, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 900, 100, 100, "mean_squared_error")
    # run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
    #              500, 1800, 100, 100, "mean_squared_error")

    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_logarithmic_error")
    run_training(GruAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_logarithmic_error")

    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 20, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 20, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 20, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 20, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 20, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 20, 20, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 100, 100, 100, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 200, 100, 100, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 300, 100, 100, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 600, 100, 100, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 900, 100, 100, "mean_squared_logarithmic_error")
    run_training(LstmAgent, TRADING_PERIOD_LENGTH, TRAINING_START_INDEX,
                 500, 1800, 100, 100, "mean_squared_logarithmic_error")


if __name__ == '__main__':
    main()
