#!/usr/bin/env python3

import config
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
    resume = False

    if len(sys.argv) > 1:
        config.run_name_baseline = sys.argv[1]
        config.config_baseline["file_name_checkpoint"] = config.run_name_baseline + ".tar"
        config.config_baseline["file_name_checkpoint_best"] = config.run_name_baseline + ".best.tar"
        resume = True

    wandb.login()
    wandb.init(project = config.project_name, name = config.run_name_baseline, config = config.config_baseline, resume = resume, mode = config.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + config.run_name_baseline + "\".")

    accuracy_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterion = torch.nn.CrossEntropyLoss()
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((wandb.config.model_input_height, wandb.config.model_input_width)),
        torchvision.transforms.ToTensor(),
    ])
    dataset_test = torchvision.datasets.ImageFolder(root = wandb.config.dir_dataset_test, transform = dataset_transforms)
    dataset_train = torchvision.datasets.ImageFolder(root = wandb.config.dir_dataset_train, transform = dataset_transforms)
    dataset_validation = torchvision.datasets.ImageFolder(root = wandb.config.dir_dataset_validation, transform = dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = wandb.config.data_loader_batch_size, shuffle = False, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model = network.BaselineNetworkA(dataset_train)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = wandb.config.optimizer_learning_rate, momentum = wandb.config.optimizer_momentum, weight_decay = wandb.config.optimizer_weight_decay)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, wandb.config.learning_rate_scheduler_mode, wandb.config.learning_rate_scheduler_factor, wandb.config.learning_rate_scheduler_patient, wandb.config.learning_rate_scheduler_threshold, wandb.config.learning_rate_scheduler_threshold_mode, wandb.config.learning_rate_scheduler_cooldown, wandb.config.learning_rate_scheduler_min_learning_rate, wandb.config.learning_rate_scheduler_min_learning_rate_decay, wandb.config.learning_rate_scheduler_verbose)
    progress_bar = None

    (accuracy_validation_best, batch_step_train, batch_step_validate, criterion, epoch) = utility.loadCheckpoint(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterion, epoch, learning_rate_scheduler, model, optimizer)

    if config.log_level >= type.LogLevel.trace:
        model_input_size = (wandb.config.model_input_channels, wandb.config.model_input_height, wandb.config.model_input_width)
        torchsummary.summary(model, input_size = model_input_size)

    if epoch <= wandb.config.epochs:
        progress_bar = tqdm.tqdm(total = wandb.config.epochs, position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= wandb.config.epochs:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train.trainBaseline(model, data_loader_train, criterion, optimizer, device, batch_step_train)
        (accuracy_validation_epoch, loss_validation_epoch, batch_step_validate) = validate.validateBaseline(model, data_loader_validation, criterion, device, batch_step_validate)

        learning_rate_scheduler.step(loss_validation_epoch)

        if accuracy_validation_epoch > accuracy_validation_best:
            accuracy_validation_best = accuracy_validation_epoch
            wandb.log({"validation/epoch/accuracy_best": accuracy_validation_best})
            utility.saveCheckpoint(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, accuracy_validation_best, batch_step_train, batch_step_validate, criterion, epoch, learning_rate_scheduler, model, optimizer)

        utility.saveCheckpoint(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterion, epoch, learning_rate_scheduler, model, optimizer)

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation accuracy: " + str(accuracy_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_best"] = accuracy_validation_best

    wandb.log({"testing/epoch/step": 1})
    batch_step_test = test.testBaseline(model, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
