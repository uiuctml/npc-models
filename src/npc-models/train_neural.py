#!/usr/bin/env python3

import argparse
import dataset
import header
import logger
import math
import model
import test_neural
import torch
import tqdm
import utility
import wandb

def computeLoss(output, labels, criterions):
    count_attributes = len(output)
    loss_attribute = 0

    for i in range(count_attributes):
        loss = criterions[i](output[i], labels[i])
        loss_attribute += loss / math.log(output[i].size(1))

    loss_attribute /= count_attributes

    return loss_attribute

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type = int, default = None, help = "Batch size.")
    parser.add_argument("-e", "--epochs", type = int, default = None, help = "Epochs.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Random seed.")
    arguments = parser.parse_args()

    if arguments.batch_size is not None:
        header.config_neural["batch_size"] = arguments.batch_size

    if arguments.epochs is not None:
        header.config_neural["epochs"] = arguments.epochs

    if arguments.seed is not None:
        header.config_neural["seed"] = arguments.seed

    test_neural.initializeRunName()

    logger.log_trace("Run name: \"" + header.config_neural["run_name"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_neural["batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_neural["epochs"]) + ".")
    logger.log_trace("Random seed: " + str(header.config_neural["seed"]) + ".")

    return

def train(model_decomposed, data_loader, criterions, optimizer, device, batch_step):
    accuracy_attribute_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_decomposed.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    with torch.set_grad_enabled(True):
        for (batch_index, (input, labels, _, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)

            for i in range(len(labels)):
                labels[i] = labels[i].to(device, non_blocking = True)

            optimizer.zero_grad()

            (output, _) = model_decomposed(input)

            loss = computeLoss(output, labels, criterions)

            loss.backward()
            optimizer.step()

            accuracy_attribute_batch = utility.computeAccuracyAttribute(output, labels, device)
            loss_batch = loss.item()

            accuracy_attribute_epoch += accuracy_attribute_batch
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"training/batch/accuracy_attribute": accuracy_attribute_batch})
            wandb.log({"training/batch/step": batch_step})
            wandb.log({"training/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    loss_epoch /= len(data_loader)

    wandb.log({"training/epoch/accuracy_attribute": accuracy_attribute_epoch})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def validate(model_decomposed, data_loader, criterions, device, batch_step):
    accuracy_attribute_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels, _, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)

            for i in range(len(labels)):
                labels[i] = labels[i].to(device, non_blocking = True)

            (output, _) = model_decomposed(input)

            loss = computeLoss(output, labels, criterions)
            accuracy_attribute_batch = utility.computeAccuracyAttribute(output, labels, device)
            loss_batch = loss.item()

            accuracy_attribute_epoch += accuracy_attribute_batch
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"validation/batch/accuracy_attribute": accuracy_attribute_batch})
            wandb.log({"validation/batch/step": batch_step})
            wandb.log({"validation/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    loss_epoch /= len(data_loader)

    wandb.log({"validation/epoch/accuracy_attribute": accuracy_attribute_epoch})
    wandb.log({"validation/epoch/loss": loss_epoch})

    return (accuracy_attribute_epoch, loss_epoch, batch_step)

def main():
    processArguments()

    utility.setSeed(header.config_neural["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.config_neural["run_name"], config = header.config_neural, mode = header.run_mode)
    utility.defineMetrics()
    logger.log_info("Started run \"" + header.config_neural["run_name"] + "\".")

    accuracy_attribute_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterions = []
    dataset_transforms = utility.createTransform(header.config_neural)
    dataset_test = dataset.NPCDataset(header.config_neural["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.NPCDataset(header.config_neural["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.NPCDataset(header.config_neural["dir_dataset_validation"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_neural["batch_size"], shuffle = False, num_workers = header.config_neural["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_neural["batch_size"], shuffle = header.config_neural["data_loader_shuffle"], num_workers = header.config_neural["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_neural["batch_size"], shuffle = header.config_neural["data_loader_shuffle"], num_workers = header.config_neural["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model_decomposed = model.ResNet34MTL(dataset_test.config, device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    optimizer = torch.optim.SGD(model_decomposed.module.get_parameters(), lr = header.config_neural["optimizer_learning_rate"], momentum = header.config_neural["optimizer_momentum"], weight_decay = header.config_neural["optimizer_weight_decay"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.config_neural["learning_rate_scheduler_mode"], header.config_neural["learning_rate_scheduler_factor"], header.config_neural["learning_rate_scheduler_patience"], header.config_neural["learning_rate_scheduler_threshold"], header.config_neural["learning_rate_scheduler_threshold_mode"], header.config_neural["learning_rate_scheduler_cooldown"], header.config_neural["learning_rate_scheduler_min_learning_rate"], header.config_neural["learning_rate_scheduler_min_learning_rate_decay"])
    progress_bar = None

    for _ in dataset_test.config["attributes"]:
        criterions.append(torch.nn.CrossEntropyLoss())

    if epoch <= header.config_neural["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_neural["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_neural["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_decomposed, data_loader_train, criterions, optimizer, device, batch_step_train)
        (accuracy_attribute_validation_epoch, loss_validation_epoch, batch_step_validate) = validate(model_decomposed, data_loader_validation, criterions, device, batch_step_validate)

        learning_rate_scheduler.step(loss_validation_epoch)

        logger.log_info("Epoch validation attribute accuracy: " + str(accuracy_attribute_validation_epoch) + ".")

        if accuracy_attribute_validation_epoch > accuracy_attribute_validation_best or epoch == 1:
            accuracy_attribute_validation_best = accuracy_attribute_validation_epoch
            wandb.log({"validation/epoch/accuracy_attribute_best": accuracy_attribute_validation_best})
            utility.saveCheckpoint(header.config_neural["file_name_checkpoint_best"], model_decomposed)

        utility.saveCheckpoint(header.config_neural["file_name_checkpoint"], model_decomposed)

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation attribute accuracy: " + str(accuracy_attribute_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_attribute_best"] = accuracy_attribute_validation_best

    wandb.log({"testing/epoch/step": batch_step_test})
    test_neural.test(model_decomposed, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
