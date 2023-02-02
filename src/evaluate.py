#!/usr/bin/env python3

import header
import json
import logger
import os
import sys
import torch
import torchmetrics.classification
import utility

def computeMeanAveragePrecision(statistics):
    (outputs_test, class_indices_test, classes) = utility.loadEvaluation(header.evaluate_model_dir)

    class_indices_test_excluded_removed = []
    class_indices_test_flatten = []
    class_indices_test_name = []
    class_indices_test_reindexed = []
    classes_excluded_removed = []
    outputs_test_excluded_removed = []
    outputs_test_flatten = []

    for output in outputs_test:
        outputs_test_flatten += output

    for class_index in class_indices_test:
        class_indices_test_flatten += class_index

    for batch_index in range(0, len(outputs_test_flatten)):
        if classes[class_indices_test_flatten[batch_index]] not in header.evaluate_label_excluded:
            outputs_test_excluded_removed.append(outputs_test_flatten[batch_index])
            class_indices_test_excluded_removed.append(class_indices_test_flatten[batch_index])

    for batch_index in range(0, len(outputs_test_flatten)):
        if classes[class_indices_test_flatten[batch_index]] in header.evaluate_label_excluded:
            for output_test_excluded_removed in outputs_test_excluded_removed:
                del output_test_excluded_removed[class_indices_test_flatten[batch_index]]

    for class_index in class_indices_test_excluded_removed:
        class_indices_test_name.append(classes[class_index])

    for class_name in classes:
        if class_name not in header.evaluate_label_excluded:
            classes_excluded_removed.append(class_name)

    for class_name in class_indices_test_name:
        class_indices_test_reindexed.append(classes_excluded_removed.index(class_name))

    outputs_test = torch.tensor(outputs_test_excluded_removed)
    class_indices_test = torch.tensor(class_indices_test_reindexed)

    metric_average_precision = torchmetrics.classification.MulticlassAveragePrecision(num_classes = len(classes_excluded_removed), average = None, thresholds = None)
    metric_mean_average_precision = torchmetrics.classification.MulticlassAveragePrecision(num_classes = len(classes_excluded_removed), average = "macro", thresholds = None)

    average_precision = metric_average_precision(outputs_test, class_indices_test)
    mean_average_precision = metric_mean_average_precision(outputs_test, class_indices_test)

    for class_index in range(0, len(classes_excluded_removed)):
        statistics[classes_excluded_removed[class_index]] = average_precision[class_index].item()

    logger.log_info("Mean average precision: " + str(mean_average_precision.item()) + ".")

    return

def main():
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        header.evaluate_model_dir = sys.argv[1]

    logger.log_info("Evaluating model in \"" + header.evaluate_model_dir + "\".")

    model_file_path_statistics_evaluate = os.path.join(header.evaluate_model_dir, header.model_file_name_statistics_evaluate)
    model_file_path_statistics_test = os.path.join(header.evaluate_model_dir, header.model_file_name_statistics_test)
    model_file_path_statistics_train = os.path.join(header.evaluate_model_dir, header.model_file_name_statistics_train)
    statistics = {}
    statistics_evaluate = None
    statistics_test = None
    statistics_train = None

    computeMeanAveragePrecision(statistics)
    utility.saveEvaluation(header.evaluate_model_dir, statistics)

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
