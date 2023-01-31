#!/usr/bin/env python3

import header
import logger
import math
import torch
import torch.nn
import torch.optim
import torchsummary
import torchvision
import tqdm
import utility

def test(model, data_loader, device, predictions):
    accuracies = []
    batch_count = math.ceil(len(data_loader.dataset) / header.test_data_loader_batch_size)
    running_corrects = 0
    progress_bar_accuracy = tqdm.tqdm(total = 1, position = 3, leave = False)
    progress_bar_progress = tqdm.tqdm(total = batch_count, position = 1, leave = False)
    progress_bar_accuracy.set_description_str("[INFO]: Validation accuracy")
    progress_bar_progress.set_description_str("[INFO]: Validation progress")

    model.eval()

    with torch.no_grad():
        for (i, (input, labels)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output = model(input)
                (_, predictions) = torch.max(output, 1)

            corrects = torch.sum(predictions == labels.data).item()
            accuracy_value = corrects / header.test_data_loader_batch_size

            running_corrects += corrects

            progress_bar_accuracy.n = round(accuracy_value, 4)
            progress_bar_progress.n = i + 1

            progress_bar_accuracy.refresh()
            progress_bar_progress.refresh()

            accuracies.append(accuracy_value)

    predictions["accuracies"] = accuracies

    progress_bar_accuracy.close()
    progress_bar_progress.close()

    return running_corrects / len(data_loader.dataset)

def main():
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.test_model_input_height, header.test_model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(root = header.dataset_dir_test, transform = dataset_transforms)
    device = torch.device("cuda")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = header.test_data_loader_batch_size, shuffle = header.test_data_loader_shuffle, num_workers = header.test_data_loader_worker_count, pin_memory = True)   
    model = torchvision.models.resnet152(weights = header.test_model_pretrained_weights)
    predictions = {}

    model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    utility.load(header.test_model_dir, model, None, None)
    logger.log_info_raw("\n")

    if header.log_level >= logger.LogLevel.debug:
        model_input_size = (header.test_model_input_channels, header.test_model_input_height, header.test_model_input_width)
        torchsummary.summary(model, input_size = model_input_size)

    utility.viewDataset(dataset, data_loader)

    accuracy = test(model, data_loader, device, predictions)

    logger.log_info("Testing accuracy: " + str(accuracy) + ".")

    return

if __name__ == "__main__":
    main()
