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
        if run_name.split(".")[2] != header.config_baseline["type"]:
            logger.log_fatal("Not a baseline run name. Quit.")
            exit(-1)
        elif run_name.split(".")[1] != header.dataset_prefix:
            logger.log_fatal("Baseline run name dataset is not \"" + header.dataset_prefix + "\". Quit.")
            exit(-1)

        header.config_baseline["run_name"] = run_name
    else:
        header.config_baseline["run_name"] = utility.generateRunName(header.config_baseline["seed"], header.config_baseline["type"], header.config_baseline["model"])

    if header.config_baseline["run_name"] == "":
        logger.log_fatal("Missing baseline run name. Quit.")
        exit(-1)

    header.config_baseline["file_name_checkpoint"] = header.config_baseline["run_name"] + header.checkpoint_postfix
    header.config_baseline["file_name_checkpoint_best"] = header.config_baseline["run_name"] + header.checkpoint_postfix_best

    return

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Seed.")
    arguments = parser.parse_args()

    initializeRunName(arguments.run_name)
    header.config_baseline["model"] = header.config_baseline["run_name"].split(".")[3]

    if arguments.seed is not None:
        header.config_baseline["seed"] = arguments.seed

    logger.log_trace("Run name: \"" + header.config_baseline["run_name"] + "\".")
    logger.log_trace("Model: \"" + header.config_baseline["model"] + "\".")
    logger.log_trace("Seed: " + str(header.config_baseline["seed"]) + ".")

    return

def computeAccuracy(output_neck, output_head, labels_attribute, labels_class, device):
    accuracy_concept_batch = None
    accuracy_classification_batch = None
    threshold_accuracy_concept = 0.5
    threshold_accuracy_classification = 0.5

    if header.config_baseline["model"] == type.ModelBaseline.cbm.name:
        threshold_accuracy_concept = 0
        threshold_accuracy_classification = 0
    elif header.config_baseline["model"] == type.ModelBaseline.abm.name or header.config_baseline["model"] == type.ModelBaseline.cem.name:
        threshold_accuracy_classification = 0

    if header.config_baseline["model"] == type.ModelBaseline.abm.name:
        accuracy_concept_batch = utility.computeConceptAccuracy(output_neck, labels_attribute, device)
    else:
        labels_attribute = utility.getBinaryLabelsAttribute(labels_attribute)
        accuracy_concept_batch = sklearn.metrics.accuracy_score(labels_attribute.cpu(), (output_neck > threshold_accuracy_concept).cpu())

    accuracy_classification_batch = sklearn.metrics.accuracy_score(labels_class.cpu(), (output_head > threshold_accuracy_classification).cpu())

    return (accuracy_concept_batch, accuracy_classification_batch)

def computeTVDistance(output_neck, labels_attribute, tv_distances_epoch, data_loader):
    counts_categories = []
    output = output_neck

    for attribute in data_loader.dataset.config["attributes"]:
        counts_categories.append(len(attribute["labels"]))

    if header.config_baseline["model"] == type.ModelBaseline.abm.name:
        output = utility.applySoftmaxAttribute(output_neck)
    elif header.config_baseline["model"] == type.ModelBaseline.cbm.name:
        output = torch.nn.functional.sigmoid(output_neck)
        output = list(torch.split(output, counts_categories, dim = 1))

        for i in range(len(output)):
            sum = torch.sum(output[i], dim = 1, keepdim = True)
            output[i] /= sum
    elif header.config_baseline["model"] == type.ModelBaseline.cem.name or header.config_baseline["model"] == type.ModelBaseline.dcr.name:
        output = list(torch.split(output_neck, counts_categories, dim = 1))

        for i in range(len(output)):
            sum = torch.sum(output[i], dim = 1, keepdim = True)
            output[i] /= sum
    else:
        return

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distance_batch = 0.5 * torch.sum(torch.abs(output[i] - labels_attribute[i]), dim = 1)
        tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

    return

def test(model_baseline, data_loader, device, batch_step):
    utility.loadCheckpoint(header.config_baseline["file_name_checkpoint_best"], model_baseline)

    accuracy_concept_epoch = 0
    accuracy_classification_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    tv_distance_epoch = 0
    tv_distances_epoch = []

    for _ in data_loader.dataset.config["attributes"]:
        tv_distances_epoch.append(0)

    model_baseline.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_attribute, labels_class, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_class = labels_class.to(device, non_blocking = True)
            labels_class = utility.getBinaryLabelsClass(labels_class, data_loader)

            for i in range(len(labels_attribute)):
                labels_attribute[i] = labels_attribute[i].to(device)

            (output_neck, output_head) = model_baseline(input)

            (accuracy_concept_batch, accuracy_classification_batch) = computeAccuracy(output_neck, output_head, labels_attribute, labels_class, device)

            accuracy_concept_epoch += accuracy_concept_batch
            accuracy_classification_epoch += accuracy_classification_batch

            computeTVDistance(output_neck, labels_attribute, tv_distances_epoch, data_loader)

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy_concept": accuracy_concept_batch})
            wandb.log({"testing/batch/accuracy_classification": accuracy_classification_batch})
            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    accuracy_concept_epoch /= len(data_loader)
    accuracy_classification_epoch /= len(data_loader)

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distances_epoch[i] /= len(data_loader.dataset)

    tv_distance_epoch = sum(tv_distances_epoch) / len(tv_distances_epoch)

    wandb.log({"testing/epoch/accuracy_concept": accuracy_concept_epoch})
    wandb.log({"testing/epoch/accuracy_classification": accuracy_classification_epoch})
    wandb.log({"testing/epoch/tv_distance": tv_distance_epoch})

    wandb.summary["testing/epoch/accuracy_concept"] = accuracy_concept_epoch
    wandb.summary["testing/epoch/accuracy_classification"] = accuracy_classification_epoch
    wandb.summary["testing/epoch/tv_distance"] = tv_distance_epoch

    logger.log_info("Testing mean TV distance: " + str(tv_distance_epoch) + ".")
    logger.log_info("Testing mean concept accuracy: " + str(accuracy_concept_epoch) + ".")
    logger.log_info("Testing classification accuracy: " + str(accuracy_classification_epoch) + ".")

    return batch_step

def main():
    processArguments()

    utility.setSeed(header.config_baseline["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_baseline, mode = "disabled")

    dataset_transforms = utility.createTransforms(header.config_baseline)
    dataset_test = dataset.NPCDataset(header.config_baseline["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_baseline["batch_size"], shuffle = False, num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_baseline = model.createModelBaseline(dataset_test.config, device)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)

    test(model_baseline, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
