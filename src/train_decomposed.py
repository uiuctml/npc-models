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
    resume = False

    if len(sys.argv) > 1:
        config.run_name_decomposed = sys.argv[1]
        config.config_decomposed["file_name_checkpoint"] = config.run_name_decomposed + ".tar"
        config.config_decomposed["file_name_checkpoint_best"] = config.run_name_decomposed + ".best.tar"
        resume = True

    if config.run_mode == "online":
        wandb.login()

    wandb.init(project = config.project_name, name = config.run_name_decomposed, config = config.config_decomposed, resume = resume, mode = config.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + config.run_name_decomposed + "\".")

    accuracy_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterions = []
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config.config_decomposed["model_input_height"], config.config_decomposed["model_input_width"])),
        torchvision.transforms.ToTensor(),
    ])
    dataset = dset.DatasetGenerated(root = config.config_decomposed["dir_dataset"], transform = dataset_transforms)
    dataset_split_lengths = [config.config_decomposed["dataset_split_percentage_test"], config.config_decomposed["dataset_split_percentage_train"], config.config_decomposed["dataset_split_percentage_validation"]]
    (dataset_subset_test, dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset, dataset_split_lengths)
    data_loader_test = torch.utils.data.DataLoader(dataset_subset_test, batch_size = config.config_decomposed["data_loader_batch_size"], shuffle = config.config_decomposed["data_loader_shuffle"], num_workers = config.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = config.config_decomposed["data_loader_batch_size"], shuffle = config.config_decomposed["data_loader_shuffle"], num_workers = config.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = config.config_decomposed["data_loader_batch_size"], shuffle = config.config_decomposed["data_loader_shuffle"], num_workers = config.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model = network.DecomposedNetworkA(dataset)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = config.config_decomposed["optimizer_learning_rate"], momentum = config.config_decomposed["optimizer_momentum"], weight_decay = config.config_decomposed["optimizer_weight_decay"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, config.config_decomposed["learning_rate_scheduler_mode"], config.config_decomposed["learning_rate_scheduler_factor"], config.config_decomposed["learning_rate_scheduler_patient"], config.config_decomposed["learning_rate_scheduler_threshold"], config.config_decomposed["learning_rate_scheduler_threshold_mode"], config.config_decomposed["learning_rate_scheduler_cooldown"], config.config_decomposed["learning_rate_scheduler_min_learning_rate"], config.config_decomposed["learning_rate_scheduler_min_learning_rate_decay"], config.config_decomposed["learning_rate_scheduler_verbose"])
    progress_bar = None

    for _ in dataset.config["datasets"]:
        criterions.append(torch.nn.CrossEntropyLoss())

    (accuracy_validation_best, batch_step_train, batch_step_validate, criterions, data_loader_test, data_loader_train, data_loader_validation, epoch) = utility.loadCheckpoint(config.config_decomposed["dir_checkpoints"], config.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, data_loader_test, data_loader_train, data_loader_validation, epoch, learning_rate_scheduler, model, optimizer)

    if config.log_level >= type.LogLevel.trace:
        model_input_size = (config.config_decomposed["model_input_channels"], config.config_decomposed["model_input_height"], config.config_decomposed["model_input_width"])
        torchsummary.summary(model, input_size = model_input_size)

    if epoch <= config.config_decomposed["epochs"]:
        progress_bar = tqdm.tqdm(total = config.config_decomposed["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= config.config_decomposed["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train.trainDecomposed(model, dataset, data_loader_train, criterions, optimizer, device, batch_step_train)
        (accuracy_validation_epoch_list, loss_overall_validation_epoch, batch_step_validate) = validate.validateDecomposed(model, dataset, data_loader_validation, criterions, device, batch_step_validate)

        learning_rate_scheduler.step(loss_overall_validation_epoch)

        accuracy_validation_epoch_mean = sum(accuracy_validation_epoch_list) / len(accuracy_validation_epoch_list)

        if accuracy_validation_epoch_mean > accuracy_validation_best:
            accuracy_validation_best = accuracy_validation_epoch_mean
            wandb.log({"validation/epoch/accuracy_best": accuracy_validation_best})
            utility.saveCheckpoint(config.config_decomposed["dir_checkpoints"], config.config_decomposed["file_name_checkpoint_best"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, data_loader_test, data_loader_train, data_loader_validation, epoch, learning_rate_scheduler, model, optimizer)

        utility.saveCheckpoint(config.config_decomposed["dir_checkpoints"], config.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, data_loader_test, data_loader_train, data_loader_validation, epoch, learning_rate_scheduler, model, optimizer)

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation accuracy: " + str(accuracy_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_best"] = accuracy_validation_best

    wandb.log({"testing/epoch/step": 1})
    batch_step_test = test.testDecomposed(model, dataset, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
