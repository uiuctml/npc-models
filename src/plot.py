#!/usr/bin/env python3

import header
import json
import logger
import os
import sys
import utility

def main():
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        header.plot_model_dir = sys.argv[1]

        if header.plot_model_dir[-1] == "/":
            header.plot_model_dir = header.plot_model_dir[:-1]

    if not os.path.isdir(header.plot_model_dir):
        logger.log_error("Invalid model directory \"" + header.plot_model_dir + "\".")
        return

    logger.log_info("Plotting model in \"" + header.plot_model_dir + "\".")

    model_file_path_statistics_evaluate = os.path.join(header.plot_model_dir, header.model_file_name_statistics_evaluate)
    model_file_path_statistics_test = os.path.join(header.plot_model_dir, header.model_file_name_statistics_test)
    model_file_path_statistics_train = os.path.join(header.plot_model_dir, header.model_file_name_statistics_train)
    statistics_evaluate = None
    statistics_test = None
    statistics_train = None

    if os.path.isfile(model_file_path_statistics_train):
        with open(model_file_path_statistics_train, "r") as file_statistics_train:
            statistics_train = json.load(file_statistics_train)
            utility.plotTrainingStatistics(statistics_train)
    else:
        logger.log_warn("Training statistics file missing.")

    if os.path.isfile(model_file_path_statistics_test):
        with open(model_file_path_statistics_test, "r") as file_statistics_test:
            statistics_test = json.load(file_statistics_test)
            utility.plotTestingStatistics(statistics_test)
    else:
        logger.log_warn("Testing statistics file missing.")

    if os.path.isfile(model_file_path_statistics_evaluate):
        with open(model_file_path_statistics_evaluate, "r") as file_statistics_evaluate:
            statistics_evaluate = json.load(file_statistics_evaluate)
            utility.plotEvaluationStatistics(statistics_evaluate)
    else:
        logger.log_warn("Evaluation statistics file missing.")

    return

if __name__ == "__main__":
    main()
