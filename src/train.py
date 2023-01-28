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

def train(model, data_loader, epoch, criterion, optimizer, device, statistics):
    accuracies = []
    batch_count = math.ceil(len(data_loader.dataset) / header.data_loader_batch_size)
    losses = []
    running_loss = 0
    running_corrects = 0
    progress_bar_accuracy = tqdm.tqdm(total = 1, position = 3, leave = False)
    progress_bar_loss = tqdm.tqdm(total = 10, position = 2, leave = False)
    progress_bar_progress = tqdm.tqdm(total = batch_count, position = 1, leave = False)
    progress_bar_accuracy.set_description_str("[INFO]: Training accuracy")
    progress_bar_loss.set_description_str("[INFO]: Training loss")
    progress_bar_progress.set_description_str("[INFO]: Training progress")

    model.train()

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

        corrects = torch.sum(predictions == labels.data).item()
        accuracy_value = corrects / header.data_loader_batch_size
        loss_value = loss.item()

        running_loss += loss_value
        running_corrects += corrects

        progress_bar_accuracy.n = accuracy_value
        progress_bar_loss.n = loss_value
        progress_bar_progress.n = i + 1

        progress_bar_accuracy.refresh()
        progress_bar_loss.refresh()
        progress_bar_progress.refresh()

        accuracies.append(accuracy_value)
        losses.append(loss_value)

    progress_bar_accuracy.close()
    progress_bar_loss.close()
    progress_bar_progress.close()

    statistics["epoch_" + str(epoch)]["batch_" + str(i)]["training_accuracies"] = accuracies
    statistics["epoch_" + str(epoch)]["batch_" + str(i)]["training_losses"] = losses

    return (running_loss, running_corrects)

def validate(model, data_loader, epoch, criterion, device, statistics):
    accuracies = []
    batch_count = math.ceil(len(data_loader.dataset) / header.data_loader_batch_size)
    losses = []
    running_loss = 0
    running_corrects = 0
    progress_bar_accuracy = tqdm.tqdm(total = 1, position = 3, leave = False)
    progress_bar_loss = tqdm.tqdm(total = 10, position = 2, leave = False)
    progress_bar_progress = tqdm.tqdm(total = batch_count, position = 1, leave = False)
    progress_bar_accuracy.set_description_str("[INFO]: Validation accuracy")
    progress_bar_loss.set_description_str("[INFO]: Validation loss")
    progress_bar_progress.set_description_str("[INFO]: Validation progress")

    model.eval()

    for (i, (input, labels)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            output = model(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)

        corrects = torch.sum(predictions == labels.data).item()
        accuracy_value = corrects / header.data_loader_batch_size
        loss_value = loss.item()

        running_loss += loss_value
        running_corrects += corrects

        progress_bar_accuracy.n = accuracy_value
        progress_bar_loss.n = loss_value
        progress_bar_progress.n = i + 1

        progress_bar_accuracy.refresh()
        progress_bar_loss.refresh()
        progress_bar_progress.refresh()

        accuracies.append(accuracy_value)
        losses.append(loss_value)

    progress_bar_accuracy.close()
    progress_bar_loss.close()
    progress_bar_progress.close()

    statistics["epoch_" + str(epoch)]["batch_" + str(i)]["validation_accuracies"] = accuracies
    statistics["epoch_" + str(epoch)]["batch_" + str(i)]["validation_losses"] = losses

    return (running_loss, running_corrects)

def main():
    accuracy_validation = None
    criterion = None
    data_loader_train = None
    data_loader_validation = None
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.model_input_height, header.model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    device = torch.device("cuda")
    epoch = None
    statistics = None

    dataset = torchvision.datasets.ImageFolder(root = header.dataset_dir_images, transform = dataset_transforms)

    model = torchvision.models.resnet152(weights = header.model_pretrained_weights)
    model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = header.optimizer_learning_rate, momentum = header.optimizer_momentum)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.learning_rate_scheduler_mode, header.learning_rate_scheduler_factor, header.learning_rate_scheduler_patient, header.learning_rate_scheduler_threshold, header.learning_rate_scheduler_threshold_mode, header.learning_rate_scheduler_cooldown, header.learning_rate_scheduler_min_learning_rate, header.learning_rate_scheduler_min_learning_rate_decay, verbose = header.learning_rate_scheduler_verbose)

    if not header.dry_run:
        (data_loader_train, data_loader_validation, epoch, criterion, accuracy_validation, statistics) = utility.load(model, optimizer, learning_rate_scheduler)
        logger.log_info_raw("\n")

    if accuracy_validation == None:
        accuracy_validation = 0

    if criterion == None:
        criterion = torch.nn.CrossEntropyLoss()

    if data_loader_train == None or data_loader_validation == None:
        dataset_split_lengths = [header.dataset_split_percentage_train, header.dataset_split_percentage_validation]
        (dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset, dataset_split_lengths)

        data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = header.data_loader_batch_size, shuffle = header.data_loader_shuffle, num_workers = header.data_loader_worker_count, pin_memory = True)
        data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = header.data_loader_batch_size, shuffle = header.data_loader_shuffle, num_workers = header.data_loader_worker_count, pin_memory = True)

    if epoch == None:
        epoch = 1

    if statistics == None:
        statistics = {}

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
        statistics_epoch_train = (0, 0)
        statistics_epoch_validation = (0, 0)

        if not header.dry_run:
            statistics["epoch_" + str(epoch)] = {}
            statistics_epoch_train = train(model, data_loader_train, epoch, criterion, optimizer, device, statistics)
            statistics_epoch_validation = validate(model, data_loader_validation, epoch, criterion, device, statistics)

        batch_count_train = math.ceil(len(data_loader_train.dataset) / header.data_loader_batch_size)
        batch_count_validation = math.ceil(len(data_loader_validation.dataset) / header.data_loader_batch_size)
        epoch_loss_train = statistics_epoch_train[0] / batch_count_train
        epoch_loss_validation = statistics_epoch_validation[0] / batch_count_validation
        epoch_accuracy_train = statistics_epoch_train[1] / len(data_loader_train.dataset)
        epoch_accuracy_validation = statistics_epoch_validation[1] / len(data_loader_validation.dataset)

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

        if not header.dry_run:
            utility.save(model, data_loader_train, data_loader_validation, epoch, criterion, optimizer, learning_rate_scheduler, accuracy_validation, statistics)

        logger.log_info_raw("\n")

    progress_bar.close()

    logger.log_info("Highest validation accuracy: " + str(accuracy_validation) + ".")

    return

if __name__ == "__main__":
    main()
