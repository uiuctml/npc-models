#!/usr/bin/env python3

import argument
import dataset
import header
import logger
import math
import model
import test_dl
import torch
import torchinfo
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

            accuracy_attribute_batch = utility.computeAccuracyDecomposed(output, labels, device)
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
            accuracy_attribute_batch = utility.computeAccuracyDecomposed(output, labels, device)
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
    resume = argument.processArgumentsTrainDecomposed()

    utility.setSeed(header.config_decomposed["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.run_name_decomposed, config = header.config_decomposed, resume = resume, mode = header.run_mode)
    utility.wAndBDefineMetrics()
    logger.log_info("Started run \"" + header.run_name_decomposed + "\".")

    accuracy_attribute_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterions = []
    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.NPCDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.NPCDataset(header.config_decomposed["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.NPCDataset(header.config_decomposed["dir_dataset_validation"], dataset_transforms)
    config_dataset = dataset_train.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    optimizer = torch.optim.SGD(model_decomposed.module.get_parameters(), lr = header.config_decomposed["optimizer_learning_rate"], momentum = header.config_decomposed["optimizer_momentum"], weight_decay = header.config_decomposed["optimizer_weight_decay"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.config_decomposed["learning_rate_scheduler_mode"], header.config_decomposed["learning_rate_scheduler_factor"], header.config_decomposed["learning_rate_scheduler_patience"], header.config_decomposed["learning_rate_scheduler_threshold"], header.config_decomposed["learning_rate_scheduler_threshold_mode"], header.config_decomposed["learning_rate_scheduler_cooldown"], header.config_decomposed["learning_rate_scheduler_min_learning_rate"], header.config_decomposed["learning_rate_scheduler_min_learning_rate_decay"])
    progress_bar = None

    for _ in config_dataset["attributes"]:
        criterions.append(torch.nn.CrossEntropyLoss())

    (accuracy_attribute_validation_best, batch_step_train, batch_step_validate, criterions, epoch) = utility.loadCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_attribute_validation_best, batch_step_train, batch_step_validate, criterions, epoch, [learning_rate_scheduler], model_decomposed, [optimizer])

    if header.show_model_summary:
        model_input_size = (header.config_decomposed["model_input_channels"], header.config_decomposed["model_input_height"], header.config_decomposed["model_input_width"])
        torchinfo.summary(model_decomposed, input_size = model_input_size)

    if epoch <= header.config_decomposed["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_decomposed["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_decomposed["epochs"]:
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
            utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], accuracy_attribute_validation_best, batch_step_train, batch_step_validate, criterions, epoch, [learning_rate_scheduler], model_decomposed, [optimizer])

        utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_attribute_validation_best, batch_step_train, batch_step_validate, criterions, epoch, [learning_rate_scheduler], model_decomposed, [optimizer])

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation attribute accuracy: " + str(accuracy_attribute_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_attribute_best"] = accuracy_attribute_validation_best

    wandb.log({"testing/epoch/step": batch_step_test})
    test_dl.test(model_decomposed, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
