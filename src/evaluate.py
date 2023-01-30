#!/usr/bin/env python3

import header
import json
import logger
import utility

def main():
    statistics = None

    with open(header.model_file_path_statistics, "r") as file_statistics:
        statistics = json.load(file_statistics)

    utility.plotStatistics(statistics)

    return

if __name__ == "__main__":
    main()
