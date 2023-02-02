#!/usr/bin/env python3

import header
import logger
import math
import os
import sys
import torch
import torch.nn
import torch.optim
import torchsummary
import torchvision
import tqdm
import utility

def test(model, data_loader, device, statistics):
    accuracies = []
    batch_count = math.ceil(len(data_loader.dataset) / header.test_data_loader_batch_size)
    class_indices = []
    running_corrects = 0
    outputs = []
    progress_bar_accuracy = tqdm.tqdm(total = 1, position = 1, leave = False)
    progress_bar_progress = tqdm.tqdm(total = batch_count, position = 0, leave = False)
    progress_bar_accuracy.set_description_str("[INFO]: Testing accuracy")
    progress_bar_progress.set_description_str("[INFO]: Testing progress")

    model.eval()

    with torch.no_grad():
        for (i, (input, labels)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output = model(input)
                (_, predictions) = torch.max(output, 1)

            corrects = torch.sum(predictions == labels.data).item()
            accuracy_value = corrects / input.size(0)

            running_corrects += corrects

            progress_bar_accuracy.n = round(accuracy_value, 4)
            progress_bar_progress.n = i + 1

            progress_bar_accuracy.refresh()
            progress_bar_progress.refresh()

            accuracies.append(accuracy_value)
            class_indices.append(labels.data.tolist())
            outputs.append(output.tolist())

    statistics["testing_accuracies"] = accuracies

    progress_bar_accuracy.close()
    progress_bar_progress.close()

    return (running_corrects, outputs, class_indices)

def main():
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        header.test_model_dir = sys.argv[1]

    if not os.path.isdir(header.test_model_dir):
        logger.log_error("Invalid model directory \"" + header.test_model_dir + "\".")
        return

    logger.log_info("Testing model in \"" + header.test_model_dir + "\".")

    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.test_model_input_height, header.test_model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(root = header.dataset_dir_test, transform = dataset_transforms)
    device = torch.device("cuda")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = header.test_data_loader_batch_size, shuffle = header.test_data_loader_shuffle, num_workers = header.test_data_loader_worker_count, pin_memory = True)   
    model = torchvision.models.resnet152(weights = header.test_model_pretrained_weights)
    statistics = {}

    model.fc = torch.nn.Sequential(torch.nn.Dropout(p = header.train_dropout_probability), torch.nn.Linear(model.fc.in_features, len(dataset.classes)))
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    utility.loadTesting(header.test_model_dir, model)

    if header.log_level >= logger.LogLevel.debug:
        model_input_size = (header.test_model_input_channels, header.test_model_input_height, header.test_model_input_width)
        torchsummary.summary(model, input_size = model_input_size)

    utility.viewDataset(dataset, data_loader)

    (running_corrects, outputs, class_indices) = test(model, data_loader, device, statistics)

    accuracy = running_corrects / len(data_loader.dataset)

    logger.log_info("Testing accuracy: " + str(accuracy) + ".")

    utility.saveTesting(header.test_model_dir, outputs, class_indices, dataset.classes, statistics)

    return

if __name__ == "__main__":
    main()
