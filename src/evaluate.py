#!/usr/bin/env python3

import header
import json
import os
import utility

def main():
    model_file_path_statistics_train = os.path.join(header.evaluate_model_dir, header.model_file_name_statistics_train)
    statistics = None

    with open(model_file_path_statistics_train, "r") as file_statistics_train:
        statistics = json.load(file_statistics_train)

    utility.plotTrainingStatistics(statistics)

    return

if __name__ == "__main__":
    main()
