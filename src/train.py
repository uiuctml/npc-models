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

def train(model, data_loader, criterion, optimizer, device):
    running_loss = 0
    running_corrects = 0

    model.train()

    batch_count = math.ceil(len(data_loader.dataset) / header.data_loader_batch_size)
    progress_bar_accuracy = tqdm.tqdm(total = 1, position = 3, leave = False)
    progress_bar_loss = tqdm.tqdm(total = 10, position = 2, leave = False)
    progress_bar_progress = tqdm.tqdm(total = batch_count, position = 1, leave = False)
    progress_bar_accuracy.set_description_str("[INFO]: Training accuracy")
    progress_bar_loss.set_description_str("[INFO]: Training loss")
    progress_bar_progress.set_description_str("[INFO]: Training progress")

    for (i, (input, labels)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        corrects = torch.sum(predictions == labels.data)

        running_loss += loss.item()
        running_corrects += corrects.item()

        progress_bar_accuracy.n = corrects.item() / header.data_loader_batch_size
        progress_bar_loss.n = loss.item()
        progress_bar_progress.n = i + 1

        progress_bar_accuracy.refresh()
        progress_bar_loss.refresh()
        progress_bar_progress.refresh()

    progress_bar_accuracy.close()
    progress_bar_loss.close()
    progress_bar_progress.close()

    return (running_loss, running_corrects)

def validate(model, data_loader, criterion, device):
    running_loss = 0
    running_corrects = 0

    model.eval()

    batch_count = math.ceil(len(data_loader.dataset) / header.data_loader_batch_size)
    progress_bar_accuracy = tqdm.tqdm(total = 1, position = 3, leave = False)
    progress_bar_loss = tqdm.tqdm(total = 10, position = 2, leave = False)
    progress_bar_progress = tqdm.tqdm(total = batch_count, position = 1, leave = False)
    progress_bar_accuracy.set_description_str("[INFO]: Validation accuracy")
    progress_bar_loss.set_description_str("[INFO]: Validation loss")
    progress_bar_progress.set_description_str("[INFO]: Validation progress")

    for (i, (input, labels)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            output = model(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)

        corrects = torch.sum(predictions == labels.data)

        running_loss += loss.item()
        running_corrects += corrects.item()

        progress_bar_accuracy.n = corrects.item() / header.data_loader_batch_size
        progress_bar_loss.n = loss.item()
        progress_bar_progress.n = i + 1

        progress_bar_accuracy.refresh()
        progress_bar_loss.refresh()
        progress_bar_progress.refresh()

    progress_bar_accuracy.close()
    progress_bar_loss.close()
    progress_bar_progress.close()

    return (running_loss, running_corrects)

def main():
    device = torch.device("cuda")

    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.model_input_height, header.model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(root = header.dataset_dir_images, transform = dataset_transforms)

    model = torchvision.models.resnet152(weights = header.model_pretrained_weights)
    model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = header.optimizer_learning_rate, momentum = header.optimizer_momentum)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.learning_rate_scheduler_mode, header.learning_rate_scheduler_factor, header.learning_rate_scheduler_patient, header.learning_rate_scheduler_threshold, header.learning_rate_scheduler_threshold_mode, header.learning_rate_scheduler_cooldown, header.learning_rate_scheduler_min_learning_rate, header.learning_rate_scheduler_min_learning_rate_decay, verbose = header.learning_rate_scheduler_verbose)

    (data_loader_train, data_loader_validation, epoch, criterion, accuracy_validation) = utility.load(model, optimizer, learning_rate_scheduler)
    logger.log_info_raw("\n")

    if data_loader_train == None or data_loader_validation == None:
        dataset_split_lengths = [header.dataset_split_percentage_train, header.dataset_split_percentage_validation]
        (dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset, dataset_split_lengths)

        data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = header.data_loader_batch_size, shuffle = header.data_loader_shuffle, num_workers = header.data_loader_worker_count, pin_memory = True)
        data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = header.data_loader_batch_size, shuffle = header.data_loader_shuffle, num_workers = header.data_loader_worker_count, pin_memory = True)

    if epoch == None:
        epoch = 1

    if criterion == None:
        criterion = torch.nn.CrossEntropyLoss()

    if accuracy_validation == None:
        accuracy_validation = 0

    if header.log_level >= logger.LogLevel.debug:
        model_input_size = (header.model_input_channels, header.model_input_height, header.model_input_width)
        torchsummary.summary(model, input_size=model_input_size)

    utility.viewDataset(dataset, data_loader_train)
    utility.viewDataset(dataset, data_loader_validation)

    progress_bar = tqdm.tqdm(total = header.model_epochs, position = 0)
    progress_bar.set_description_str("[INFO]: Epoch")
    progress_bar.n = epoch
    progress_bar.refresh()

    while epoch <= header.model_epochs:
        stats_train = (0, 0)
        stats_validation = (0, 0)

        if not header.dry_run:
            stats_train = train(model, data_loader_train, criterion, optimizer, device)
            stats_validation = validate(model, data_loader_validation, criterion, device)

        epoch_loss_train = stats_train[0] / header.data_loader_batch_size
        epoch_loss_validation = stats_validation[0] / header.data_loader_batch_size
        epoch_accuracy_train = stats_train[1] / len(data_loader_train.dataset)
        epoch_accuracy_validation = stats_validation[1] / len(data_loader_validation.dataset)

        progress_bar.n = epoch
        progress_bar.refresh()

        logger.log_info("Training loss: " + str(epoch_loss_train) + ".")
        logger.log_info("Validation loss: " + str(epoch_loss_validation) + ".")
        logger.log_info("Training accuracy: " + str(epoch_accuracy_train) + ".")
        logger.log_info("Validation accuracy: " + str(epoch_accuracy_validation) + ".")

        learning_rate_scheduler.step(epoch_loss_validation)
        epoch += 1

        if epoch_accuracy_validation > accuracy_validation:
            accuracy_validation = epoch_accuracy_validation

        utility.save(model, data_loader_train, data_loader_validation, epoch, criterion, optimizer, learning_rate_scheduler, accuracy_validation)
        logger.log_info_raw("\n")

    progress_bar.close()

    logger.log_info("Highest validation accuracy: " + str(accuracy_validation) + ".")

    return

if __name__ == "__main__":
    main()
