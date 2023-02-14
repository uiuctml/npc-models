#!/usr/bin/env python3

import header
import logger
import math
import network
import torch
import torch.nn
import torch.optim
import torchsummary
import torchvision
import tqdm
import type
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

def train(model, data_loader, epoch, criterion, optimizer, device, statistics):
    accuracies = []
    batch_count = math.ceil(len(data_loader.dataset) / header.train_data_loader_batch_size)
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

            if header.train_use_l2_loss:
                l2_norm = utility.computeL2Norm(model.parameters())
                loss_l2 = header.train_l2_lambda * l2_norm
                loss += loss_l2

            loss.backward()
            optimizer.step()

        corrects = torch.sum(predictions == labels.data).item()
        accuracy_value = corrects / input.size(0)
        loss_value = loss.item()

        running_loss += loss_value
        running_corrects += corrects

        progress_bar_accuracy.n = round(accuracy_value, 4)
        progress_bar_loss.n = round(loss_value, 4)
        progress_bar_progress.n = i + 1

        progress_bar_accuracy.refresh()
        progress_bar_loss.refresh()
        progress_bar_progress.refresh()

        accuracies.append(accuracy_value)
        losses.append(loss_value)

    progress_bar_accuracy.close()
    progress_bar_loss.close()
    progress_bar_progress.close()

    statistics["epoch_" + str(epoch)]["training_accuracies"] = accuracies
    statistics["epoch_" + str(epoch)]["training_losses"] = losses

    return (running_loss, running_corrects)

def validate(model, data_loader, epoch, criterion, device, statistics):
    accuracies = []
    batch_count = math.ceil(len(data_loader.dataset) / header.train_data_loader_batch_size)
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
        accuracy_value = corrects / input.size(0)
        loss_value = loss.item()

        running_loss += loss_value
        running_corrects += corrects

        progress_bar_accuracy.n = round(accuracy_value, 4)
        progress_bar_loss.n = round(loss_value, 4)
        progress_bar_progress.n = i + 1

        progress_bar_accuracy.refresh()
        progress_bar_loss.refresh()
        progress_bar_progress.refresh()

        accuracies.append(accuracy_value)
        losses.append(loss_value)

    progress_bar_accuracy.close()
    progress_bar_loss.close()
    progress_bar_progress.close()

    statistics["epoch_" + str(epoch)]["validation_accuracies"] = accuracies
    statistics["epoch_" + str(epoch)]["validation_losses"] = losses

    return (running_loss, running_corrects)

def main():
    accuracy_validation = None
    criterion = None
    data_loader_train = None
    data_loader_validation = None
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.train_model_input_height, header.train_model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    device = torch.device("cuda")
    epoch = None
    statistics_test = {}
    statistics_train = None

    dataset_test = torchvision.datasets.ImageFolder(root = header.dataset_dir_test, transform = dataset_transforms)
    dataset_train_validation = torchvision.datasets.ImageFolder(root = header.dataset_dir_train_validation, transform = dataset_transforms)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.test_data_loader_batch_size, shuffle = header.test_data_loader_shuffle, num_workers = header.test_data_loader_worker_count, pin_memory = True)

    model = torchvision.models.resnet152(weights = header.train_model_pretrained_weights)

    if header.train_load_best:
        utility.loadTrainingBest(header.train_model_dir_best, model)

    if not header.train_fine_tuning:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model = network.configure(header.train_network_revision, model, dataset_train_validation, device)

    optimizer = torch.optim.SGD(model.parameters(), lr = header.train_optimizer_learning_rate, momentum = header.train_optimizer_momentum, weight_decay = header.train_optimizer_weight_decay)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.train_learning_rate_scheduler_mode, header.train_learning_rate_scheduler_factor, header.train_learning_rate_scheduler_patient, header.train_learning_rate_scheduler_threshold, header.train_learning_rate_scheduler_threshold_mode, header.train_learning_rate_scheduler_cooldown, header.train_learning_rate_scheduler_min_learning_rate, header.train_learning_rate_scheduler_min_learning_rate_decay, header.train_learning_rate_scheduler_verbose)

    if not header.train_dry_run:
        (data_loader_train, data_loader_validation, epoch, criterion, accuracy_validation, statistics_train) = utility.loadTraining(header.train_model_dir, model, optimizer, learning_rate_scheduler)
        logger.log_info_raw("\n")

    if accuracy_validation == None:
        accuracy_validation = 0

    if criterion == None:
        criterion = torch.nn.CrossEntropyLoss()

    if data_loader_train == None or data_loader_validation == None:
        dataset_split_lengths = [header.dataset_split_percentage_train, header.dataset_split_percentage_validation]
        (dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset_train_validation, dataset_split_lengths)

        data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = header.train_data_loader_batch_size, shuffle = header.train_data_loader_shuffle, num_workers = header.train_data_loader_worker_count, pin_memory = True)
        data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = header.train_data_loader_batch_size, shuffle = header.train_data_loader_shuffle, num_workers = header.train_data_loader_worker_count, pin_memory = True)

    if epoch == None:
        epoch = 1

    if statistics_train == None:
        statistics_train = {}

    if header.log_level >= type.LogLevel.debug:
        model_input_size = (header.train_model_input_channels, header.train_model_input_height, header.train_model_input_width)
        torchsummary.summary(model, input_size = model_input_size)

    utility.viewDataset(dataset_train_validation, data_loader_train)
    utility.viewDataset(dataset_train_validation, data_loader_validation)
    utility.viewDataset(dataset_test, data_loader_test)

    progress_bar = tqdm.tqdm(total = header.train_epochs, position = 0)
    progress_bar.set_description_str("[INFO]: Epoch")
    progress_bar.n = epoch
    progress_bar.refresh()

    while epoch <= header.train_epochs:
        best = False
        statistics_epoch_train = (0, 0)
        statistics_epoch_validation = (0, 0)

        if not header.train_dry_run:
            statistics_train["epoch_" + str(epoch)] = {}
            statistics_epoch_train = train(model, data_loader_train, epoch, criterion, optimizer, device, statistics_train)
            statistics_epoch_validation = validate(model, data_loader_validation, epoch, criterion, device, statistics_train)

        batch_count_train = math.ceil(len(data_loader_train.dataset) / header.train_data_loader_batch_size)
        batch_count_validation = math.ceil(len(data_loader_validation.dataset) / header.train_data_loader_batch_size)
        epoch_loss_train = statistics_epoch_train[0] / batch_count_train
        epoch_loss_validation = statistics_epoch_validation[0] / batch_count_validation
        epoch_accuracy_train = statistics_epoch_train[1] / len(data_loader_train.dataset)
        epoch_accuracy_validation = statistics_epoch_validation[1] / len(data_loader_validation.dataset)

        logger.log_info("Training loss: " + str(epoch_loss_train) + ".")
        logger.log_info("Validation loss: " + str(epoch_loss_validation) + ".")
        logger.log_info("Training accuracy: " + str(epoch_accuracy_train) + ".")
        logger.log_info("Validation accuracy: " + str(epoch_accuracy_validation) + ".")

        learning_rate_scheduler.step(epoch_loss_validation)
        epoch += 1

        if epoch_accuracy_validation > accuracy_validation:
            accuracy_validation = epoch_accuracy_validation
            best = True

        if not header.train_dry_run:
            utility.saveTraining(header.train_model_dir, model, data_loader_train, data_loader_validation, epoch, criterion, optimizer, learning_rate_scheduler, accuracy_validation, statistics_train, best)

        logger.log_info_raw("\n")

        if epoch <= header.train_epochs:
            progress_bar.n = epoch
            progress_bar.refresh()

    progress_bar.close()

    logger.log_info("Highest validation accuracy: " + str(accuracy_validation) + ".")

    statistics_epoch_test = (0, [], [])

    logger.log_info("Testing model in \"" + header.test_model_dir + "\".")

    if not header.train_dry_run:
        statistics_epoch_test = test(model, data_loader_test, device, statistics_test)

    accuracy_test = statistics_epoch_test[0] / len(data_loader_test.dataset)

    logger.log_info("Testing accuracy: " + str(accuracy_test) + ".")

    if not header.train_dry_run:
        utility.saveTesting(header.test_model_dir, statistics_epoch_test[1], statistics_epoch_test[2], dataset_test.classes, statistics_test)

    return

if __name__ == "__main__":
    main()
