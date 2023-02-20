
import header
import torch
import torchmetrics.classification

def evaluateBaseline(outputs, class_indices, classes, statistics):
    class_indices_test_excluded_removed = []
    class_indices_test_flatten = []
    class_indices_test_name = []
    class_indices_test_reindexed = []
    classes_excluded_removed = []
    outputs_test_excluded_removed = []
    outputs_test_flatten = []

    for output in outputs:
        outputs_test_flatten += output

    for class_index in class_indices:
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

    outputs = torch.tensor(outputs_test_excluded_removed)
    class_indices = torch.tensor(class_indices_test_reindexed)

    metric_average_precision = torchmetrics.classification.MulticlassAveragePrecision(num_classes = len(classes_excluded_removed), average = None, thresholds = None)
    metric_mean_average_precision = torchmetrics.classification.MulticlassAveragePrecision(num_classes = len(classes_excluded_removed), average = "macro", thresholds = None)

    average_precision = metric_average_precision(outputs, class_indices)
    mean_average_precision = metric_mean_average_precision(outputs, class_indices)

    for class_index in range(0, len(classes_excluded_removed)):
        statistics[classes_excluded_removed[class_index]] = average_precision[class_index].item()

    return mean_average_precision
