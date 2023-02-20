#!/usr/bin/env python3

import header
import logger
import network
import torch
import torch.nn
import torch.optim
import torchsummary
import torchvision
import test
import tqdm
import train
import type
import utility
import validate

def main():
    accuracy_validation = None
    criterion = None
    data_loader_test = None
    data_loader_train = None
    data_loader_validation = None
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.baseline_model_input_height, header.baseline_model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    device = torch.device("cuda")
    epoch = None
    progress_bar = None
    statistics_test = {}
    statistics_train = None

    dataset_test = torchvision.datasets.ImageFolder(root = header.dataset_dir_test, transform = dataset_transforms)
    dataset_train_validation = torchvision.datasets.ImageFolder(root = header.dataset_dir_train_validation, transform = dataset_transforms)

    model = network.BaselineNetworkA(dataset_train_validation)

    if header.baseline_load_best:
        utility.loadTrainingBest(header.baseline_model_dir_best, model)

    if not header.baseline_fine_tuning:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = header.baseline_optimizer_learning_rate, momentum = header.baseline_optimizer_momentum, weight_decay = header.baseline_optimizer_weight_decay)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.baseline_learning_rate_scheduler_mode, header.baseline_learning_rate_scheduler_factor, header.baseline_learning_rate_scheduler_patient, header.baseline_learning_rate_scheduler_threshold, header.baseline_learning_rate_scheduler_threshold_mode, header.baseline_learning_rate_scheduler_cooldown, header.baseline_learning_rate_scheduler_min_learning_rate, header.baseline_learning_rate_scheduler_min_learning_rate_decay, header.baseline_learning_rate_scheduler_verbose)

    if not header.baseline_dry_run:
        (data_loader_test, data_loader_train, data_loader_validation, epoch, criterion, accuracy_validation, statistics_train) = utility.loadTraining(header.baseline_model_dir, model, optimizer, learning_rate_scheduler)
        logger.log_info_raw("\n")

    if accuracy_validation == None:
        accuracy_validation = 0

    if criterion == None:
        criterion = torch.nn.CrossEntropyLoss()

    if data_loader_train == None or data_loader_validation == None:
        dataset_split_lengths = [header.baseline_dataset_split_percentage_train, header.baseline_dataset_split_percentage_validation]
        (dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset_train_validation, dataset_split_lengths)

        data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = header.baseline_data_loader_batch_size, shuffle = header.baseline_data_loader_shuffle, num_workers = header.baseline_data_loader_worker_count, pin_memory = True)
        data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = header.baseline_data_loader_batch_size, shuffle = header.baseline_data_loader_shuffle, num_workers = header.baseline_data_loader_worker_count, pin_memory = True)

    if data_loader_test == None:
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.baseline_data_loader_batch_size, shuffle = header.baseline_data_loader_shuffle, num_workers = header.baseline_data_loader_worker_count, pin_memory = True)

    if epoch == None:
        epoch = 1

    if statistics_train == None:
        statistics_train = {}

    if header.log_level >= type.LogLevel.debug:
        model_input_size = (header.baseline_model_input_channels, header.baseline_model_input_height, header.baseline_model_input_width)
        torchsummary.summary(model, input_size = model_input_size)

    utility.viewDatasetBaseline(dataset_train_validation, data_loader_train, header.baseline_data_loader_batch_size)
    utility.viewDatasetBaseline(dataset_train_validation, data_loader_validation, header.baseline_data_loader_batch_size)
    utility.viewDatasetBaseline(dataset_test, data_loader_test, header.baseline_data_loader_batch_size)

    if epoch <= header.baseline_epochs:
        progress_bar = tqdm.tqdm(total = header.baseline_epochs, position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")
        progress_bar.n = epoch
        progress_bar.refresh()

    while epoch <= header.baseline_epochs:
        best = False
        statistics_epoch_train = (0, 0)
        statistics_epoch_validation = (0, 0)

        if not header.baseline_dry_run:
            statistics_train["epoch_" + str(epoch)] = {}
            statistics_epoch_train = train.trainBaseline(model, data_loader_train, epoch, criterion, optimizer, device, statistics_train)
            statistics_epoch_validation = validate.validateBaseline(model, data_loader_validation, epoch, criterion, device, statistics_train)

        epoch_loss_train = statistics_epoch_train[0] / len(data_loader_train)
        epoch_loss_validation = statistics_epoch_validation[0] / len(data_loader_validation)
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

        if not header.baseline_dry_run:
            utility.saveTraining(header.baseline_model_dir, model, data_loader_test, data_loader_train, data_loader_validation, epoch, criterion, optimizer, learning_rate_scheduler, accuracy_validation, statistics_train, best)

        logger.log_info_raw("\n")

        if progress_bar != None and epoch <= header.baseline_epochs:
            progress_bar.n = epoch
            progress_bar.refresh()

    if progress_bar != None:
        progress_bar.close()

    logger.log_info("Highest validation accuracy: " + str(accuracy_validation) + ".")
    logger.log_info_raw("\n")

    statistics_epoch_test = (0, [], [])

    utility.loadTesting(header.baseline_model_dir, model)

    logger.log_info("Testing best model in \"" + header.baseline_model_dir + "\".")

    if not header.baseline_dry_run:
        statistics_epoch_test = test.testBaseline(model, data_loader_test, device, statistics_test)

    accuracy_test = statistics_epoch_test[0] / len(data_loader_test.dataset)

    logger.log_info("Testing accuracy: " + str(accuracy_test) + ".")

    if not header.baseline_dry_run:
        utility.saveTesting(header.baseline_model_dir, statistics_epoch_test[1], statistics_epoch_test[2], dataset_test.classes, statistics_test)

    return

if __name__ == "__main__":
    main()
