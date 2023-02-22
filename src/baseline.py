#!/usr/bin/env python3

import config
import evaluate
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
import wandb

def main():
    wandb.login()
    wandb.init(project = config.project_name, name = config.run_name_baseline, config = config.config_baseline)

    accuracy_validation = None
    criterion = None
    data_loader_test = None
    data_loader_train = None
    data_loader_validation = None
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((wandb.config.model_input_height, wandb.config.model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    device = torch.device("cuda")
    epoch = None
    progress_bar = None
    statistics_evaluate = {}
    statistics_test = {}
    statistics_train = None

    dataset = torchvision.datasets.ImageFolder(root = wandb.config.dataset_dir, transform = dataset_transforms)
    model = network.BaselineNetworkA(dataset)

    if wandb.config.load_best:
        utility.loadTrainingBest(wandb.config.model_dir_best, model)

    if not wandb.config.fine_tuning:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = wandb.config.optimizer_learning_rate, momentum = wandb.config.optimizer_momentum, weight_decay = wandb.config.optimizer_weight_decay)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, wandb.config.learning_rate_scheduler_mode, wandb.config.learning_rate_scheduler_factor, wandb.config.learning_rate_scheduler_patient, wandb.config.learning_rate_scheduler_threshold, wandb.config.learning_rate_scheduler_threshold_mode, wandb.config.learning_rate_scheduler_cooldown, wandb.config.learning_rate_scheduler_min_learning_rate, wandb.config.learning_rate_scheduler_min_learning_rate_decay, wandb.config.learning_rate_scheduler_verbose)

    if not wandb.config.dry_run:
        (data_loader_test, data_loader_train, data_loader_validation, epoch, criterion, accuracy_validation, statistics_train) = utility.loadTraining(wandb.config.model_dir, model, optimizer, learning_rate_scheduler)
        logger.log_info_raw("\n")

    if accuracy_validation == None:
        accuracy_validation = 0

    if criterion == None:
        criterion = torch.nn.CrossEntropyLoss()

    if data_loader_train == None or data_loader_validation == None or data_loader_test == None:
        dataset_split_lengths = [wandb.config.dataset_split_percentage_test, wandb.config.dataset_split_percentage_train, wandb.config.dataset_split_percentage_validation]
        (dataset_subset_test, dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset, dataset_split_lengths)

        data_loader_test = torch.utils.data.DataLoader(dataset_subset_test, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
        data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
        data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)

    if epoch == None:
        epoch = 1

    if statistics_train == None:
        statistics_train = {}

    if config.log_level >= type.LogLevel.debug:
        model_input_size = (wandb.config.model_input_channels, wandb.config.model_input_height, wandb.config.model_input_width)
        torchsummary.summary(model, input_size = model_input_size)

    if epoch <= wandb.config.epochs:
        progress_bar = tqdm.tqdm(total = wandb.config.epochs, position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")
        progress_bar.n = epoch
        progress_bar.refresh()

    while epoch <= wandb.config.epochs:
        best = False
        statistics_epoch_train = (0, 0)
        statistics_epoch_validation = (0, 0)

        if not wandb.config.dry_run:
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

        if not wandb.config.dry_run:
            utility.saveTraining(wandb.config.model_dir, model, data_loader_test, data_loader_train, data_loader_validation, epoch, criterion, optimizer, learning_rate_scheduler, accuracy_validation, statistics_train, best)

        logger.log_info_raw("\n")

        if progress_bar is not None and epoch <= wandb.config.epochs:
            progress_bar.n = epoch
            progress_bar.refresh()

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Highest validation accuracy: " + str(accuracy_validation) + ".")
    logger.log_info_raw("\n")

    statistics_epoch_test = (0, [], [])

    utility.loadTesting(wandb.config.model_dir, model)

    logger.log_info("Testing best model in \"" + wandb.config.model_dir + "\".")

    if not wandb.config.dry_run:
        statistics_epoch_test = test.testBaseline(model, data_loader_test, device, statistics_test)

    accuracy_test = statistics_epoch_test[0] / len(data_loader_test.dataset)

    logger.log_info("Testing accuracy: " + str(accuracy_test) + ".")

    if not wandb.config.dry_run:
        utility.saveTesting(wandb.config.model_dir, statistics_epoch_test[1], statistics_epoch_test[2], dataset.classes, statistics_test)

    (outputs_test, class_indices_test, classes) = utility.loadEvaluation(wandb.config.model_dir)

    logger.log_info("Evaluating model in \"" + wandb.config.model_dir + "\".")

    mean_average_precision = evaluate.evaluateBaseline(outputs_test, class_indices_test, classes, statistics_evaluate)

    logger.log_info("Mean average precision: " + str(mean_average_precision.item()) + ".")

    wandb.finish()

    return

if __name__ == "__main__":
    main()
