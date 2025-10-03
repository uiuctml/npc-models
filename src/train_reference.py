#!/usr/bin/env python3

import argparse
import dataset
import header
import logger
import math
import model
import test_reference
import torch
import torchinfo
import tqdm
import type
import utility
import wandb

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type = str, default = "", help = "Model.")
    parser.add_argument("-b", "--batch-size", type = int, default = None, help = "Batch size.")
    parser.add_argument("-e", "--epochs", type = int, default = None, help = "Epochs.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Random seed.")
    arguments = parser.parse_args()

    if arguments.model != "":
        header.config_reference["model"] = arguments.model

    if arguments.batch_size is not None:
        header.config_reference["data_loader_batch_size"] = arguments.batch_size

    if arguments.epochs is not None:
        header.config_reference["epochs"] = arguments.epochs

    if arguments.seed is not None:
        header.config_reference["seed"] = arguments.seed

    test_reference.initializeRunName()

    logger.log_trace("Run name: \"" + header.run_name_reference + "\".")
    logger.log_trace("Model: \"" + header.config_reference["model"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_reference["data_loader_batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_reference["epochs"]) + ".")
    logger.log_trace("Random seed: " + str(header.config_reference["seed"]) + ".")

    return

def computeLoss(output_neck, output_head, labels_decomposed, labels_original):
    if header.config_reference["model"] == type.ModelReference.cbm.name:
        labels_decomposed = utility.getBinaryLabelsDecomposed(labels_decomposed)
        loss_attribute = torch.nn.functional.binary_cross_entropy_with_logits(output_neck, labels_decomposed)
        loss_task = torch.nn.functional.binary_cross_entropy_with_logits(output_head, labels_original)
        return header.config_reference["concept_loss_weight"] * loss_attribute + loss_task
    elif header.config_reference["model"] == type.ModelReference.cbm_cat.name:
        count_attributes = len(output_neck)
        loss_attribute = 0

        for i in range(count_attributes):
            loss = torch.nn.functional.cross_entropy(output_neck[i], labels_decomposed[i])
            loss_attribute += loss / math.log(output_neck[i].size(1))

        loss_attribute /= count_attributes
        loss_task = torch.nn.functional.binary_cross_entropy_with_logits(output_head, labels_original)

        return header.config_reference["concept_loss_weight"] * loss_attribute + loss_task
    elif header.config_reference["model"] == type.ModelReference.cem.name:
        labels_decomposed = utility.getBinaryLabelsDecomposed(labels_decomposed)
        loss_attribute = torch.nn.functional.binary_cross_entropy(output_neck, labels_decomposed)
        loss_task = torch.nn.functional.binary_cross_entropy_with_logits(output_head, labels_original)
        return header.config_reference["concept_loss_weight"] * loss_attribute + loss_task
    elif header.config_reference["model"] == type.ModelReference.dcr.name:
        labels_decomposed = utility.getBinaryLabelsDecomposed(labels_decomposed)
        loss_attribute = torch.nn.functional.binary_cross_entropy(output_neck, labels_decomposed)
        loss_task = torch.nn.functional.binary_cross_entropy(output_head, labels_original)
        return header.config_reference["concept_loss_weight"] * loss_attribute + loss_task
    else:
        logger.log_fatal("Unknown reference network model \"" + header.config_reference["model"] + "\".")
        exit(-1)

    return

def train(model_reference, data_loader, optimizer, device, batch_step):
    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_reference.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    with torch.set_grad_enabled(True):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)
            labels_original = utility.getBinaryLabelsOriginal(labels_original, data_loader)

            for i in range(len(labels_decomposed)):
                labels_decomposed[i] = labels_decomposed[i].to(device)

            optimizer.zero_grad()

            (output_neck, output_head) = model_reference(input)

            loss = computeLoss(output_neck, output_head, labels_decomposed, labels_original)

            loss.backward()
            optimizer.step()

            (accuracy_attribute_batch, accuracy_task_batch) = test_reference.computeAccuracy(output_neck, output_head, labels_decomposed, labels_original, device)
            loss_batch = loss.item()

            accuracy_attribute_epoch += accuracy_attribute_batch
            accuracy_task_epoch += accuracy_task_batch
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"training/batch/accuracy_attribute": accuracy_attribute_batch})
            wandb.log({"training/batch/accuracy_task": accuracy_task_batch})
            wandb.log({"training/batch/step": batch_step})
            wandb.log({"training/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    accuracy_task_epoch /= len(data_loader)
    loss_epoch /= len(data_loader)

    wandb.log({"training/epoch/accuracy_attribute": accuracy_attribute_epoch})
    wandb.log({"training/epoch/accuracy_task": accuracy_task_epoch})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def validate(model_reference, data_loader, device, batch_step):
    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_reference.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)
            labels_original = utility.getBinaryLabelsOriginal(labels_original, data_loader)

            for i in range(len(labels_decomposed)):
                labels_decomposed[i] = labels_decomposed[i].to(device)

            (output_neck, output_head) = model_reference(input)

            loss = computeLoss(output_neck, output_head, labels_decomposed, labels_original)
            (accuracy_attribute_batch, accuracy_task_batch) = test_reference.computeAccuracy(output_neck, output_head, labels_decomposed, labels_original, device)
            loss_batch = loss.item()

            accuracy_attribute_epoch += accuracy_attribute_batch
            accuracy_task_epoch += accuracy_task_batch
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"validation/batch/accuracy_attribute": accuracy_attribute_batch})
            wandb.log({"validation/batch/accuracy_task": accuracy_task_batch})
            wandb.log({"validation/batch/step": batch_step})
            wandb.log({"validation/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    accuracy_task_epoch /= len(data_loader)
    loss_epoch /= len(data_loader)

    wandb.log({"validation/epoch/accuracy_attribute": accuracy_attribute_epoch})
    wandb.log({"validation/epoch/accuracy_task": accuracy_task_epoch})
    wandb.log({"validation/epoch/loss": loss_epoch})

    return (accuracy_task_epoch, loss_epoch, batch_step)

def main():
    processArguments()

    utility.setSeed(header.config_reference["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.run_name_reference, config = header.config_reference, mode = header.run_mode)
    utility.wAndBDefineMetrics()
    logger.log_info("Started run \"" + header.run_name_reference + "\".")

    accuracy_task_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    dataset_transforms = utility.createTransform(header.config_reference)
    dataset_test = dataset.NPCDataset(header.config_reference["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.NPCDataset(header.config_reference["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.NPCDataset(header.config_reference["dir_dataset_validation"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_reference["data_loader_batch_size"], shuffle = False, num_workers = header.config_reference["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_reference["data_loader_batch_size"], shuffle = header.config_reference["data_loader_shuffle"], num_workers = header.config_reference["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_reference["data_loader_batch_size"], shuffle = header.config_reference["data_loader_shuffle"], num_workers = header.config_reference["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model_reference = model.createModelReference(dataset_test.config, device)
    model_reference = torch.nn.DataParallel(model_reference)
    model_reference = model_reference.to(device)
    optimizer = torch.optim.SGD(model_reference.module.get_parameters(), lr = header.config_reference["optimizer_learning_rate"], momentum = header.config_reference["optimizer_momentum"], weight_decay = header.config_reference["optimizer_weight_decay"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.config_reference["learning_rate_scheduler_mode"], header.config_reference["learning_rate_scheduler_factor"], header.config_reference["learning_rate_scheduler_patience"])
    progress_bar = tqdm.tqdm(total = header.config_reference["epochs"], position = 0)

    progress_bar.set_description_str("[INFO]: Epoch")

    if header.show_model_summary:
        model_input_size = (header.config_reference["model_input_channels"], header.config_reference["model_input_height"], header.config_reference["model_input_width"])
        torchinfo.summary(model_reference, input_size = model_input_size)

    while epoch <= header.config_reference["epochs"]:
        progress_bar.n = epoch
        progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_reference, data_loader_train, optimizer, device, batch_step_train)
        (accuracy_task_validation_epoch, loss_validation_epoch, batch_step_validate) = validate(model_reference, data_loader_validation, device, batch_step_validate)

        learning_rate_scheduler.step(loss_validation_epoch)

        logger.log_info("Epoch validation task accuracy: " + str(accuracy_task_validation_epoch) + ".")

        if accuracy_task_validation_epoch > accuracy_task_validation_best or epoch == 1:
            accuracy_task_validation_best = accuracy_task_validation_epoch
            wandb.log({"validation/epoch/accuracy_task_best": accuracy_task_validation_best})
            utility.saveCheckpoint(header.config_reference["dir_checkpoints"], header.config_reference["file_name_checkpoint_best"], accuracy_task_validation_best, 0, 0, [], epoch, [], model_reference, [])

        utility.saveCheckpoint(header.config_reference["dir_checkpoints"], header.config_reference["file_name_checkpoint"], accuracy_task_validation_best, 0, 0, [], epoch, [], model_reference, [])

        epoch += 1

    progress_bar.close()

    logger.log_info("Best validation task accuracy: " + str(accuracy_task_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_task_best"] = accuracy_task_validation_best

    wandb.log({"testing/epoch/step": batch_step_test})
    test_reference.test(model_reference, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
