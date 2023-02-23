#!/usr/bin/env python3

import config
import dataset as dset
import logger
import network
import sys
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
    if len(sys.argv) > 1:
        config.run_name_decomposed = sys.argv[1]
        config.config_decomposed["file_name_checkpoint"] = config.run_name_decomposed + ".tar"
        config.config_decomposed["file_name_checkpoint_best"] = config.run_name_decomposed + ".best.tar"

    wandb.login()
    wandb.init(project = config.project_name, name = config.run_name_decomposed, config = config.config_decomposed, resume = True)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + config.run_name_decomposed + "\".")

    accuracy_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterions = []
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((wandb.config.model_input_height, wandb.config.model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = dset.DatasetGenerated(root = wandb.config.dataset_dir, transform = dataset_transforms)
    dataset_split_lengths = [wandb.config.dataset_split_percentage_test, wandb.config.dataset_split_percentage_train, wandb.config.dataset_split_percentage_validation]
    (dataset_subset_test, dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset, dataset_split_lengths)
    data_loader_test = torch.utils.data.DataLoader(dataset_subset_test, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model = network.DecomposedNetworkA(dataset)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = wandb.config.optimizer_learning_rate, momentum = wandb.config.optimizer_momentum, weight_decay = wandb.config.optimizer_weight_decay)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, wandb.config.learning_rate_scheduler_mode, wandb.config.learning_rate_scheduler_factor, wandb.config.learning_rate_scheduler_patient, wandb.config.learning_rate_scheduler_threshold, wandb.config.learning_rate_scheduler_threshold_mode, wandb.config.learning_rate_scheduler_cooldown, wandb.config.learning_rate_scheduler_min_learning_rate, wandb.config.learning_rate_scheduler_min_learning_rate_decay, wandb.config.learning_rate_scheduler_verbose)
    progress_bar = None

    for _ in dataset.config["datasets"]:
        criterions.append(torch.nn.CrossEntropyLoss())

    (accuracy_validation_best, batch_step_train, batch_step_validate, criterion, data_loader_test, data_loader_train, data_loader_validation, epoch) = utility.loadCheckpoint(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterion, data_loader_test, data_loader_train, data_loader_validation, epoch, learning_rate_scheduler, model, optimizer)

    if config.log_level >= type.LogLevel.debug:
        model_input_size = (wandb.config.model_input_channels, wandb.config.model_input_height, wandb.config.model_input_width)
        torchsummary.summary(model, input_size = model_input_size)

    if epoch <= wandb.config.epochs:
        progress_bar = tqdm.tqdm(total = wandb.config.epochs, position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")
        progress_bar.n = epoch
        progress_bar.refresh()

    while epoch <= wandb.config.epochs:
        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train.trainDecomposed(model, dataset, data_loader_train, criterions, optimizer, device, batch_step_train)
        batch_step_validate = validate.validateDecomposed(model, dataset, data_loader_validation, criterions, device, batch_step_validate)

        epoch_loss_train = statistics_epoch_train[0] / len(data_loader_train)
        epoch_loss_validation = statistics_epoch_validation[0] / len(data_loader_validation)

        logger.log_info("Training loss: " + str(epoch_loss_train) + ".")

        for (i, dataset_entry) in enumerate(dataset.config["datasets"]):
            epoch_accuracy_train = statistics_epoch_train[1][i] / len(data_loader_train.dataset)
            logger.log_info("Training accuracy for \"" + dataset_entry["name"] + "\": " + str(epoch_accuracy_train) + ".")

        logger.log_info("Validation loss: " + str(epoch_loss_validation) + ".")

        for (i, dataset_entry) in enumerate(dataset.config["datasets"]):
            epoch_accuracy_validation = statistics_epoch_validation[1][i] / len(data_loader_validation.dataset)
            epoch_mean_accuracy_validation += epoch_accuracy_validation
            logger.log_info("Validation accuracy for \"" + dataset_entry["name"] + "\": " + str(epoch_accuracy_validation) + ".")

        learning_rate_scheduler.step(epoch_loss_validation)
        epoch += 1
        epoch_mean_accuracy_validation /= len(dataset.config["datasets"])

        if epoch_mean_accuracy_validation > accuracy_validation_best:
            accuracy_validation_best = epoch_mean_accuracy_validation
            best = True

        if not wandb.config.dry_run:
            utility.saveTraining(wandb.config.model_dir, model, data_loader_test, data_loader_train, data_loader_validation, epoch, criterions, optimizer, learning_rate_scheduler, accuracy_validation_best, statistics_train, best)

        logger.log_info_raw("\n")

        if progress_bar is not None and epoch <= wandb.config.epochs:
            progress_bar.n = epoch
            progress_bar.refresh()

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Highest validation accuracy: " + str(accuracy_validation_best) + ".")
    logger.log_info_raw("\n")

    statistics_epoch_test = (0, [], [])

    utility.loadTesting(wandb.config.model_dir, model)

    logger.log_info("Testing best model in \"" + wandb.config.model_dir + "\".")

    if not wandb.config.dry_run:
        statistics_epoch_test = test.testDecomposed(model, dataset, data_loader_test, device, statistics_test)

        for (i, dataset_entry) in enumerate(dataset.config["datasets"]):
                accuracy_test = statistics_epoch_test[0][i] / len(data_loader_test.dataset)
                logger.log_info("Testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_test) + ".")

        utility.saveTesting(wandb.config.model_dir, statistics_epoch_test[1], statistics_epoch_test[2], dataset.classes, statistics_test)

    (outputs_test, class_indices_test, classes) = utility.loadEvaluation(wandb.config.model_dir)

    logger.log_info("Evaluating model in \"" + wandb.config.model_dir + "\".")

    mean_average_precisions = evaluate.evaluateDecomposed(dataset, outputs_test, class_indices_test, classes, statistics_evaluate)

    for dataset_name in mean_average_precisions.keys():
        logger.log_info("Mean average precision for \"" + dataset_name + "\": " + str(mean_average_precisions[dataset_name].item()) + ".")

    wandb.finish()

    return

if __name__ == "__main__":
    main()
