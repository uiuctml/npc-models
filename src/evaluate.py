import torch
import torchmetrics.classification

def evaluateBaseline(outputs, class_indices, classes, statistics):
    outputs = torch.tensor(outputs)
    class_indices = torch.tensor(class_indices)

    metric_average_precision = torchmetrics.classification.MulticlassAveragePrecision(num_classes = len(classes), average = None, thresholds = None)
    metric_mean_average_precision = torchmetrics.classification.MulticlassAveragePrecision(num_classes = len(classes), average = "macro", thresholds = None)

    average_precision = metric_average_precision(outputs, class_indices)
    mean_average_precision = metric_mean_average_precision(outputs, class_indices)

    for class_index in range(0, len(classes)):
        statistics[classes[class_index]] = average_precision[class_index].item()

    return mean_average_precision

def evaluateDecomposed(dataset, outputs, class_indices, classes, statistics):
    mean_average_precisions = {}

    for (i, dataset_entry) in enumerate(dataset.config["datasets"]):
        dataset_name = dataset_entry["name"]
        outputs_dataset = torch.tensor(outputs[i])
        class_indices_dataset = torch.tensor(class_indices[i])

        metric_average_precision = torchmetrics.classification.MulticlassAveragePrecision(num_classes = len(classes[i]), average = None, thresholds = None)
        metric_mean_average_precision = torchmetrics.classification.MulticlassAveragePrecision(num_classes = len(classes[i]), average = "macro", thresholds = None)

        average_precision = metric_average_precision(outputs_dataset, class_indices_dataset)
        mean_average_precision = metric_mean_average_precision(outputs_dataset, class_indices_dataset)

        mean_average_precisions[dataset_name] = mean_average_precision
        statistics[dataset_name] = {}

        for class_index in range(0, len(classes[i])):
            statistics[dataset_name][classes[i][class_index]] = average_precision[class_index].item()

    return mean_average_precisions
