#!/usr/bin/env python3

import argparse
import dataset
import header
import logger
import model
import sklearn.metrics
import torch
import tqdm
import type
import utility
import wandb

def initializeRunName(run_name = ""):
    if run_name != "":
        if run_name.split(".")[1] != header.config_reference["type"]:
            logger.log_fatal("Invalid reference run name. Quit.")
            exit(-1)

        header.run_name_reference = run_name
    else:
        header.run_name_reference = utility.generateRunName(header.config_reference["type"], header.config_reference["model"], header.config_reference["seed"])

    if header.run_name_reference == "":
        logger.log_fatal("Missing reference run name. Quit.")
        exit(-1)

    header.config_reference["file_name_checkpoint"] = header.run_name_reference + ".tar"
    header.config_reference["file_name_checkpoint_best"] = header.run_name_reference + ".best.tar"
    header.config_reference["run_name"] = header.run_name_reference

    return

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Random seed.")
    arguments = parser.parse_args()

    initializeRunName(arguments.run_name)
    header.config_reference["model"] = header.run_name_reference.split(".")[2]

    if arguments.seed is not None:
        header.config_reference["seed"] = arguments.seed

    logger.log_trace("Run name: \"" + header.run_name_reference + "\".")
    logger.log_trace("Model: \"" + header.config_reference["model"] + "\".")
    logger.log_trace("Random seed: " + str(header.config_reference["seed"]) + ".")

    return

def computeAccuracy(output_neck, output_head, labels_decomposed, labels_original, device):
    accuracy_attribute_batch = None
    accuracy_task_batch = None
    threshold_accuracy_attribute = 0.5
    threshold_accuracy_task = 0.5

    if header.config_reference["model"] == type.ModelReference.cbm.name:
        threshold_accuracy_attribute = 0
        threshold_accuracy_task = 0
    elif header.config_reference["model"] == type.ModelReference.cbm_cat.name or header.config_reference["model"] == type.ModelReference.cem.name:
        threshold_accuracy_task = 0

    if header.config_reference["model"] == type.ModelReference.cbm_cat.name:
        accuracy_attribute_batch = utility.computeAccuracyDecomposed(output_neck, labels_decomposed, device)
    else:
        labels_decomposed = utility.getBinaryLabelsDecomposed(labels_decomposed)
        accuracy_attribute_batch = sklearn.metrics.accuracy_score(labels_decomposed.cpu(), (output_neck > threshold_accuracy_attribute).cpu())

    accuracy_task_batch = sklearn.metrics.accuracy_score(labels_original.cpu(), (output_head > threshold_accuracy_task).cpu())

    return (accuracy_attribute_batch, accuracy_task_batch)

def computeTVDistance(output_neck, labels_decomposed, tv_distances_epoch, data_loader):
    counts_categories = []
    output = output_neck

    for attribute in data_loader.dataset.config["attributes"]:
        counts_categories.append(len(attribute["labels"]))

    if header.config_reference["model"] == type.ModelReference.cbm.name:
        output = torch.nn.functional.sigmoid(output_neck)
        output = list(torch.split(output, counts_categories, dim = 1))

        for i in range(len(output)):
            sum = torch.sum(output[i], dim = 1, keepdim = True)
            output[i] /= sum
    elif header.config_reference["model"] == type.ModelReference.cbm_cat.name:
        output = utility.applySoftmaxDecomposed(output_neck)
    elif header.config_reference["model"] == type.ModelReference.cem.name or header.config_reference["model"] == type.ModelReference.dcr.name:
        output = list(torch.split(output_neck, counts_categories, dim = 1))

        for i in range(len(output)):
            sum = torch.sum(output[i], dim = 1, keepdim = True)
            output[i] /= sum
    else:
        return

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distance_batch = 0.5 * torch.sum(torch.abs(output[i] - labels_decomposed[i]), dim = 1)
        tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

    return

def test(model_reference, data_loader, device, batch_step):
    utility.loadCheckpoint(header.config_reference["file_name_checkpoint_best"], model_reference)

    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    tv_distance_epoch = 0
    tv_distances_epoch = []

    for _ in data_loader.dataset.config["attributes"]:
        tv_distances_epoch.append(0)

    model_reference.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)
            labels_original = utility.getBinaryLabelsOriginal(labels_original, data_loader)

            for i in range(len(labels_decomposed)):
                labels_decomposed[i] = labels_decomposed[i].to(device)

            (output_neck, output_head) = model_reference(input)

            (accuracy_attribute_batch, accuracy_task_batch) = computeAccuracy(output_neck, output_head, labels_decomposed, labels_original, device)

            accuracy_attribute_epoch += accuracy_attribute_batch
            accuracy_task_epoch += accuracy_task_batch

            computeTVDistance(output_neck, labels_decomposed, tv_distances_epoch, data_loader)

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy_attribute": accuracy_attribute_batch})
            wandb.log({"testing/batch/accuracy_task": accuracy_task_batch})
            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    accuracy_task_epoch /= len(data_loader)

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distances_epoch[i] /= len(data_loader.dataset)

    tv_distance_epoch = sum(tv_distances_epoch) / len(tv_distances_epoch)

    wandb.log({"testing/epoch/accuracy_attribute": accuracy_attribute_epoch})
    wandb.log({"testing/epoch/accuracy_task": accuracy_task_epoch})
    wandb.log({"testing/epoch/tv_distance_attribute": tv_distance_epoch})

    wandb.summary["testing/epoch/accuracy_attribute"] = accuracy_attribute_epoch
    wandb.summary["testing/epoch/accuracy_task"] = accuracy_task_epoch
    wandb.summary["testing/epoch/tv_distance_attribute"] = tv_distance_epoch

    logger.log_info("Testing attribute TV distance: " + str(tv_distance_epoch) + ".")
    logger.log_info("Testing attribute accuracy: " + str(accuracy_attribute_epoch) + ".")
    logger.log_info("Testing task accuracy: " + str(accuracy_task_epoch) + ".")

    return batch_step

def main():
    processArguments()

    utility.setSeed(header.config_reference["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_reference, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_reference)
    dataset_test = dataset.NPCDataset(header.config_reference["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_reference["batch_size"], shuffle = False, num_workers = header.config_reference["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_reference = model.createModelReference(dataset_test.config, device)
    model_reference = torch.nn.DataParallel(model_reference)
    model_reference = model_reference.to(device)

    test(model_reference, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
