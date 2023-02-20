#!/usr/bin/env python3

import dataset
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
    criterions = None
    data_loader_test = None
    data_loader_train = None
    data_loader_validation = None
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.decomposed_model_input_height, header.decomposed_model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    device = torch.device("cuda")
    epoch = None
    progress_bar = None
    statistics_test = {}
    statistics_train = None

    dataset_generated = dataset.DatasetGenerated(root = header.dataset_dir_generated, transform = dataset_transforms)

    model = network.DecomposedNetworkA(dataset_generated)

    if header.decomposed_load_best:
        utility.loadTrainingBest(header.decomposed_model_dir_best, model)

    if not header.decomposed_fine_tuning:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = header.decomposed_optimizer_learning_rate, momentum = header.decomposed_optimizer_momentum, weight_decay = header.decomposed_optimizer_weight_decay)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.decomposed_learning_rate_scheduler_mode, header.decomposed_learning_rate_scheduler_factor, header.decomposed_learning_rate_scheduler_patient, header.decomposed_learning_rate_scheduler_threshold, header.decomposed_learning_rate_scheduler_threshold_mode, header.decomposed_learning_rate_scheduler_cooldown, header.decomposed_learning_rate_scheduler_min_learning_rate, header.decomposed_learning_rate_scheduler_min_learning_rate_decay, header.decomposed_learning_rate_scheduler_verbose)

    if not header.decomposed_dry_run:
        (data_loader_test, data_loader_train, data_loader_validation, epoch, criterions, accuracy_validation, statistics_train) = utility.loadTraining(header.decomposed_model_dir, model, optimizer, learning_rate_scheduler)
        logger.log_info_raw("\n")

    if accuracy_validation == None:
        accuracy_validation = 0

    if criterions == None:
        criterions = []

        for _ in range(0, len(dataset_generated.config["datasets"])):
            criterions.append(torch.nn.CrossEntropyLoss())

    if data_loader_train == None or data_loader_validation == None or data_loader_test == None:
        dataset_split_lengths = [header.decomposed_dataset_split_percentage_test, header.decomposed_dataset_split_percentage_train, header.decomposed_dataset_split_percentage_validation]
        (dataset_subset_test, dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset_generated, dataset_split_lengths)

        data_loader_test = torch.utils.data.DataLoader(dataset_subset_test, batch_size = header.decomposed_data_loader_batch_size, shuffle = header.decomposed_data_loader_shuffle, num_workers = header.decomposed_data_loader_worker_count, pin_memory = True)
        data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = header.decomposed_data_loader_batch_size, shuffle = header.decomposed_data_loader_shuffle, num_workers = header.decomposed_data_loader_worker_count, pin_memory = True)
        data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = header.decomposed_data_loader_batch_size, shuffle = header.decomposed_data_loader_shuffle, num_workers = header.decomposed_data_loader_worker_count, pin_memory = True)

    if epoch == None:
        epoch = 1

    if statistics_train == None:
        statistics_train = {}

    if header.log_level >= type.LogLevel.debug:
        model_input_size = (header.decomposed_model_input_channels, header.decomposed_model_input_height, header.decomposed_model_input_width)
        torchsummary.summary(model, input_size = model_input_size)

    utility.viewDatasetDecomposed(dataset_generated, data_loader_train, header.decomposed_data_loader_batch_size)
    utility.viewDatasetDecomposed(dataset_generated, data_loader_validation, header.decomposed_data_loader_batch_size)
    utility.viewDatasetDecomposed(dataset_generated, data_loader_test, header.decomposed_data_loader_batch_size)

    if epoch <= header.decomposed_epochs:
        progress_bar = tqdm.tqdm(total = header.decomposed_epochs, position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")
        progress_bar.n = epoch
        progress_bar.refresh()

    while epoch <= header.decomposed_epochs:
        best = False
        epoch_loss_validation = 0
        epoch_mean_accuracy_validation = 0

        if not header.decomposed_dry_run:
            statistics_train["epoch_" + str(epoch)] = {}
            statistics_epoch_train = train.trainDecomposed(model, dataset_generated, data_loader_train, epoch, criterions, optimizer, device, statistics_train)
            statistics_epoch_validation = validate.validateDecomposed(model, dataset_generated, data_loader_validation, epoch, criterions, device, statistics_train)

            epoch_loss_train = statistics_epoch_train[0] / len(data_loader_train)
            epoch_loss_validation = statistics_epoch_validation[0] / len(data_loader_validation)

            logger.log_info("Training loss: " + str(epoch_loss_train) + ".")

            for (i, dataset_entry) in enumerate(dataset_generated.config["datasets"]):
                epoch_accuracy_train = statistics_epoch_train[1][i] / len(data_loader_train.dataset)
                logger.log_info("Training accuracy for \"" + dataset_entry["name"] + "\": " + str(epoch_accuracy_train) + ".")

            logger.log_info("Validation loss: " + str(epoch_loss_validation) + ".")

            for (i, dataset_entry) in enumerate(dataset_generated.config["datasets"]):
                epoch_accuracy_validation = statistics_epoch_validation[1][i] / len(data_loader_validation.dataset)
                epoch_mean_accuracy_validation += epoch_accuracy_validation
                logger.log_info("Validation accuracy for \"" + dataset_entry["name"] + "\": " + str(epoch_accuracy_validation) + ".")

        learning_rate_scheduler.step(epoch_loss_validation)
        epoch += 1
        epoch_mean_accuracy_validation /= len(dataset_generated.config["datasets"])

        if epoch_mean_accuracy_validation > accuracy_validation:
            accuracy_validation = epoch_mean_accuracy_validation
            best = True

        if not header.decomposed_dry_run:
            utility.saveTraining(header.decomposed_model_dir, model, data_loader_test, data_loader_train, data_loader_validation, epoch, criterions, optimizer, learning_rate_scheduler, accuracy_validation, statistics_train, best)

        logger.log_info_raw("\n")

        if progress_bar is not None and epoch <= header.decomposed_epochs:
            progress_bar.n = epoch
            progress_bar.refresh()

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Highest validation accuracy: " + str(accuracy_validation) + ".")
    logger.log_info_raw("\n")

    statistics_epoch_test = (0, [], [])

    utility.loadTesting(header.decomposed_model_dir, model)

    logger.log_info("Testing best model in \"" + header.decomposed_model_dir + "\".")

    if not header.decomposed_dry_run:
        statistics_epoch_test = test.testDecomposed(model, dataset_generated, data_loader_test, device, statistics_test)

        for (i, dataset_entry) in enumerate(dataset_generated.config["datasets"]):
                accuracy_test = statistics_epoch_test[0][i] / len(data_loader_test.dataset)
                logger.log_info("Testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_test) + ".")

        utility.saveTesting(header.decomposed_model_dir, statistics_epoch_test[1], statistics_epoch_test[2], dataset_generated.classes, statistics_test)

    return

if __name__ == "__main__":
    main()
