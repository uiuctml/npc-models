#!/usr/bin/env python3

import header
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
    run_name = ""

    if len(sys.argv) > 1:
        run_name = sys.argv[1]

    resume = utility.initializeRunNameDecomposed(run_name)

    if header.run_name_decomposed == "":
        logger.log_error("Run name missing. Quit.")
        return

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.run_name_decomposed, config = header.config_decomposed, resume = resume, mode = header.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + header.run_name_decomposed + "\".")

    accuracy_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterions = []
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.config_decomposed["model_input_height"], header.config_decomposed["model_input_width"])),
        torchvision.transforms.ToTensor(),
    ])
    dataset_original = torchvision.datasets.ImageFolder(header.config_baseline["dir_dataset_test"], dataset_transforms)
    dataset_test = dset.DatasetDecomposed(header.config_decomposed["dir_dataset_test"], dataset_original.classes, dataset_transforms)
    dataset_train = dset.DatasetDecomposed(header.config_decomposed["dir_dataset_train"], dataset_original.classes, dataset_transforms)
    dataset_validation = dset.DatasetDecomposed(header.config_decomposed["dir_dataset_validation"], dataset_original.classes, dataset_transforms)
    config_dataset = dataset_train.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model = network.createModelDecomposed(config_dataset)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = header.config_decomposed["optimizer_learning_rate"], momentum = header.config_decomposed["optimizer_momentum"], weight_decay = header.config_decomposed["optimizer_weight_decay"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.config_decomposed["learning_rate_scheduler_mode"], header.config_decomposed["learning_rate_scheduler_factor"], header.config_decomposed["learning_rate_scheduler_patient"], header.config_decomposed["learning_rate_scheduler_threshold"], header.config_decomposed["learning_rate_scheduler_threshold_mode"], header.config_decomposed["learning_rate_scheduler_cooldown"], header.config_decomposed["learning_rate_scheduler_min_learning_rate"], header.config_decomposed["learning_rate_scheduler_min_learning_rate_decay"], header.config_decomposed["learning_rate_scheduler_verbose"])
    progress_bar = None

    for _ in config_dataset["datasets"]:
        criterions.append(torch.nn.CrossEntropyLoss())

    (accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch) = utility.loadCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_scheduler, model, optimizer)

    if header.log_level >= type.LogLevel.trace:
        model_input_size = (header.config_decomposed["model_input_channels"], header.config_decomposed["model_input_height"], header.config_decomposed["model_input_width"])
        torchsummary.summary(model, input_size = model_input_size)

    if epoch <= header.config_decomposed["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_decomposed["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_decomposed["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train.trainDecomposed(model, config_dataset, data_loader_train, criterions, optimizer, device, batch_step_train)
        (accuracy_validation_epoch_list, loss_overall_validation_epoch, batch_step_validate) = validate.validateDecomposed(model, config_dataset, data_loader_validation, criterions, device, batch_step_validate)

        learning_rate_scheduler.step(loss_overall_validation_epoch)

        accuracy_validation_epoch_mean = sum(accuracy_validation_epoch_list) / len(accuracy_validation_epoch_list)

        if accuracy_validation_epoch_mean > accuracy_validation_best:
            accuracy_validation_best = accuracy_validation_epoch_mean
            wandb.log({"validation/epoch/accuracy_best": accuracy_validation_best})
            utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_scheduler, model, optimizer)

        utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_scheduler, model, optimizer)

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation accuracy: " + str(accuracy_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_best"] = accuracy_validation_best

    wandb.log({"testing/epoch/step": 1})
    batch_step_test = test.testDecomposed(model, config_dataset, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
