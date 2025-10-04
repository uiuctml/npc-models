#!/usr/bin/env python3

import argparse
import dataset
import header
import logger
import model
import test_blackbox
import torch
import tqdm
import utility
import wandb

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type = int, default = None, help = "Batch size.")
    parser.add_argument("-e", "--epochs", type = int, default = None, help = "Epochs.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Random seed.")
    arguments = parser.parse_args()

    if arguments.batch_size is not None:
        header.config_baseline["batch_size"] = arguments.batch_size

    if arguments.epochs is not None:
        header.config_baseline["epochs"] = arguments.epochs

    if arguments.seed is not None:
        header.config_baseline["seed"] = arguments.seed

    test_blackbox.initializeRunName()

    logger.log_trace("Run name: \"" + header.config_baseline["run_name"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_baseline["batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_baseline["epochs"]) + ".")
    logger.log_trace("Random seed: " + str(header.config_baseline["seed"]) + ".")

    return

def train(model_baseline, data_loader, criterion, optimizer, device, batch_step):
    accuracy_task_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_baseline.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    with torch.set_grad_enabled(True):
        for (batch_index, (input, _, labels, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            optimizer.zero_grad()

            output = model_baseline(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            corrects = torch.sum(predictions == labels).item()

            accuracy_task_batch = corrects / input.size(0)
            loss_batch = loss.item()

            accuracy_task_epoch += corrects
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"training/batch/accuracy_task": accuracy_task_batch})
            wandb.log({"training/batch/step": batch_step})
            wandb.log({"training/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_task_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"training/epoch/accuracy_task": accuracy_task_epoch})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def validate(model_baseline, data_loader, criterion, device, batch_step):
    accuracy_task_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_baseline.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, _, labels, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            output = model_baseline(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)

            corrects = torch.sum(predictions == labels).item()
            accuracy_task_batch = corrects / input.size(0)
            loss_batch = loss.item()

            accuracy_task_epoch += corrects
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"validation/batch/accuracy_task": accuracy_task_batch})
            wandb.log({"validation/batch/step": batch_step})
            wandb.log({"validation/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_task_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"validation/epoch/accuracy_task": accuracy_task_epoch})
    wandb.log({"validation/epoch/loss": loss_epoch})

    return (accuracy_task_epoch, loss_epoch, batch_step)

def main():
    processArguments()

    utility.setSeed(header.config_baseline["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.config_baseline["run_name"], config = header.config_baseline, mode = header.run_mode)

    utility.defineMetrics()

    logger.log_info("Started run \"" + header.config_baseline["run_name"] + "\".")

    accuracy_task_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterion = torch.nn.CrossEntropyLoss()
    dataset_transforms = utility.createTransform(header.config_baseline)
    dataset_test = dataset.NPCDataset(header.config_baseline["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.NPCDataset(header.config_baseline["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.NPCDataset(header.config_baseline["dir_dataset_validation"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_baseline["batch_size"], shuffle = False, num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_baseline["batch_size"], shuffle = header.config_baseline["data_loader_shuffle"], num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_baseline["batch_size"], shuffle = header.config_baseline["data_loader_shuffle"], num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model_baseline = model.ResNet34(dataset_test.config, device)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)
    optimizer = torch.optim.SGD(model_baseline.module.get_parameters(), lr = header.config_baseline["optimizer_learning_rate"], momentum = header.config_baseline["optimizer_momentum"], weight_decay = header.config_baseline["optimizer_weight_decay"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.config_baseline["learning_rate_scheduler_mode"], header.config_baseline["learning_rate_scheduler_factor"], header.config_baseline["learning_rate_scheduler_patience"], header.config_baseline["learning_rate_scheduler_threshold"], header.config_baseline["learning_rate_scheduler_threshold_mode"], header.config_baseline["learning_rate_scheduler_cooldown"], header.config_baseline["learning_rate_scheduler_min_learning_rate"], header.config_baseline["learning_rate_scheduler_min_learning_rate_decay"])
    progress_bar = None

    if epoch <= header.config_baseline["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_baseline["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_baseline["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_baseline, data_loader_train, criterion, optimizer, device, batch_step_train)
        (accuracy_task_validation_epoch, loss_validation_epoch, batch_step_validate) = validate(model_baseline, data_loader_validation, criterion, device, batch_step_validate)

        learning_rate_scheduler.step(loss_validation_epoch)

        logger.log_info("Epoch validation task accuracy: " + str(accuracy_task_validation_epoch) + ".")

        if accuracy_task_validation_epoch > accuracy_task_validation_best or epoch == 1:
            accuracy_task_validation_best = accuracy_task_validation_epoch
            wandb.log({"validation/epoch/accuracy_task_best": accuracy_task_validation_best})
            utility.saveCheckpoint(header.config_baseline["file_name_checkpoint_best"], model_baseline)

        utility.saveCheckpoint(header.config_baseline["file_name_checkpoint"], model_baseline)

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation task accuracy: " + str(accuracy_task_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_task_best"] = accuracy_task_validation_best

    wandb.log({"testing/epoch/step": batch_step_test})
    test_blackbox.test(model_baseline, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
