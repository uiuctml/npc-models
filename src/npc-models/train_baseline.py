#!/usr/bin/env python3

import argparse
import dataset
import header
import logger
import math
import model
import test_baseline
import torch
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
        header.config_baseline["model"] = arguments.model

    if arguments.batch_size is not None:
        header.config_baseline["batch_size"] = arguments.batch_size

    if arguments.epochs is not None:
        header.config_baseline["epochs"] = arguments.epochs

    if arguments.seed is not None:
        header.config_baseline["seed"] = arguments.seed

    test_baseline.initializeRunName()

    logger.log_trace("Run name: \"" + header.config_baseline["run_name"] + "\".")
    logger.log_trace("Model: \"" + header.config_baseline["model"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_baseline["batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_baseline["epochs"]) + ".")
    logger.log_trace("Random seed: " + str(header.config_baseline["seed"]) + ".")

    return

def computeLoss(output_neck, output_head, labels_attribute, labels_class):
    if header.config_baseline["model"] == type.ModelBaseline.abm.name:
        count_attributes = len(output_neck)
        loss_attribute = 0

        for i in range(count_attributes):
            loss = torch.nn.functional.cross_entropy(output_neck[i], labels_attribute[i])
            loss_attribute += loss / math.log(output_neck[i].size(1))

        loss_attribute /= count_attributes
        loss_task = torch.nn.functional.binary_cross_entropy_with_logits(output_head, labels_class)

        return header.config_baseline["concept_loss_weight"] * loss_attribute + loss_task
    elif header.config_baseline["model"] == type.ModelBaseline.cbm.name:
        labels_attribute = utility.getBinaryLabelsAttribute(labels_attribute)
        loss_attribute = torch.nn.functional.binary_cross_entropy_with_logits(output_neck, labels_attribute)
        loss_task = torch.nn.functional.binary_cross_entropy_with_logits(output_head, labels_class)
        return header.config_baseline["concept_loss_weight"] * loss_attribute + loss_task
    elif header.config_baseline["model"] == type.ModelBaseline.cem.name:
        labels_attribute = utility.getBinaryLabelsAttribute(labels_attribute)
        loss_attribute = torch.nn.functional.binary_cross_entropy(output_neck, labels_attribute)
        loss_task = torch.nn.functional.binary_cross_entropy_with_logits(output_head, labels_class)
        return header.config_baseline["concept_loss_weight"] * loss_attribute + loss_task
    elif header.config_baseline["model"] == type.ModelBaseline.dcr.name:
        labels_attribute = utility.getBinaryLabelsAttribute(labels_attribute)
        loss_attribute = torch.nn.functional.binary_cross_entropy(output_neck, labels_attribute)
        loss_task = torch.nn.functional.binary_cross_entropy(output_head, labels_class)
        return header.config_baseline["concept_loss_weight"] * loss_attribute + loss_task
    else:
        logger.log_fatal("Unknown baseline model \"" + header.config_baseline["model"] + "\".")
        exit(-1)

    return

def train(model_baseline, data_loader, optimizer, device, batch_step):
    accuracy_concept_epoch = 0
    accuracy_classification_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_baseline.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    with torch.set_grad_enabled(True):
        for (batch_index, (input, labels_attribute, labels_class, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_class = labels_class.to(device, non_blocking = True)
            labels_class = utility.getBinaryLabelsClass(labels_class, data_loader)

            for i in range(len(labels_attribute)):
                labels_attribute[i] = labels_attribute[i].to(device)

            optimizer.zero_grad()

            (output_neck, output_head) = model_baseline(input)

            loss = computeLoss(output_neck, output_head, labels_attribute, labels_class)

            loss.backward()
            optimizer.step()

            (accuracy_concept_batch, accuracy_classification_batch) = test_baseline.computeAccuracy(output_neck, output_head, labels_attribute, labels_class, device)
            loss_batch = loss.item()

            accuracy_concept_epoch += accuracy_concept_batch
            accuracy_classification_epoch += accuracy_classification_batch
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"training/batch/accuracy_concept": accuracy_concept_batch})
            wandb.log({"training/batch/accuracy_classification": accuracy_classification_batch})
            wandb.log({"training/batch/step": batch_step})
            wandb.log({"training/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_concept_epoch /= len(data_loader)
    accuracy_classification_epoch /= len(data_loader)
    loss_epoch /= len(data_loader)

    wandb.log({"training/epoch/accuracy_concept": accuracy_concept_epoch})
    wandb.log({"training/epoch/accuracy_classification": accuracy_classification_epoch})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def validate(model_baseline, data_loader, device, batch_step):
    accuracy_concept_epoch = 0
    accuracy_classification_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_baseline.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_attribute, labels_class, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_class = labels_class.to(device, non_blocking = True)
            labels_class = utility.getBinaryLabelsClass(labels_class, data_loader)

            for i in range(len(labels_attribute)):
                labels_attribute[i] = labels_attribute[i].to(device)

            (output_neck, output_head) = model_baseline(input)

            loss = computeLoss(output_neck, output_head, labels_attribute, labels_class)
            (accuracy_concept_batch, accuracy_classification_batch) = test_baseline.computeAccuracy(output_neck, output_head, labels_attribute, labels_class, device)
            loss_batch = loss.item()

            accuracy_concept_epoch += accuracy_concept_batch
            accuracy_classification_epoch += accuracy_classification_batch
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"validation/batch/accuracy_concept": accuracy_concept_batch})
            wandb.log({"validation/batch/accuracy_classification": accuracy_classification_batch})
            wandb.log({"validation/batch/step": batch_step})
            wandb.log({"validation/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_concept_epoch /= len(data_loader)
    accuracy_classification_epoch /= len(data_loader)
    loss_epoch /= len(data_loader)

    wandb.log({"validation/epoch/accuracy_concept": accuracy_concept_epoch})
    wandb.log({"validation/epoch/accuracy_classification": accuracy_classification_epoch})
    wandb.log({"validation/epoch/loss": loss_epoch})

    return (accuracy_classification_epoch, loss_epoch, batch_step)

def main():
    processArguments()

    utility.setSeed(header.config_baseline["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.config_baseline["run_name"], config = header.config_baseline, mode = header.run_mode)
    utility.defineMetrics()
    logger.log_info("Started run \"" + header.config_baseline["run_name"] + "\".")

    accuracy_classification_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    dataset_transforms = utility.createTransform(header.config_baseline)
    dataset_test = dataset.NPCDataset(header.config_baseline["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.NPCDataset(header.config_baseline["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.NPCDataset(header.config_baseline["dir_dataset_validation"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_baseline["batch_size"], shuffle = False, num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_baseline["batch_size"], shuffle = header.config_baseline["data_loader_shuffle"], num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_baseline["batch_size"], shuffle = header.config_baseline["data_loader_shuffle"], num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model_baseline = model.createModelBaseline(dataset_test.config, device)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)
    optimizer = torch.optim.SGD(model_baseline.module.get_parameters(), lr = header.config_baseline["optimizer_learning_rate"], momentum = header.config_baseline["optimizer_momentum"], weight_decay = header.config_baseline["optimizer_weight_decay"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.config_baseline["learning_rate_scheduler_mode"], header.config_baseline["learning_rate_scheduler_factor"], header.config_baseline["learning_rate_scheduler_patience"])
    progress_bar = tqdm.tqdm(total = header.config_baseline["epochs"], position = 0)

    progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_baseline["epochs"]:
        progress_bar.n = epoch
        progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_baseline, data_loader_train, optimizer, device, batch_step_train)
        (accuracy_classification_validation_epoch, loss_validation_epoch, batch_step_validate) = validate(model_baseline, data_loader_validation, device, batch_step_validate)

        learning_rate_scheduler.step(loss_validation_epoch)

        logger.log_info("Validation classification accuracy: " + str(accuracy_classification_validation_epoch) + ".")

        if accuracy_classification_validation_epoch > accuracy_classification_validation_best or epoch == 1:
            accuracy_classification_validation_best = accuracy_classification_validation_epoch
            wandb.log({"validation/epoch/accuracy_classification_best": accuracy_classification_validation_best})
            utility.saveCheckpoint(header.config_baseline["file_name_checkpoint_best"], model_baseline)

        utility.saveCheckpoint(header.config_baseline["file_name_checkpoint"], model_baseline)

        epoch += 1

    progress_bar.close()

    logger.log_info("Best validation classification accuracy: " + str(accuracy_classification_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_classification_best"] = accuracy_classification_validation_best

    wandb.log({"testing/epoch/step": batch_step_test})
    test_baseline.test(model_baseline, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
