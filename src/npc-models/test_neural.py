#!/usr/bin/env python3

import argparse
import dataset
import header
import logger
import model
import torch
import tqdm
import utility
import wandb

def initializeRunName(run_name = ""):
    if run_name != "":
        if run_name.split(".")[2] != header.config_neural["type"]:
            logger.log_fatal("Not a neural run name. Quit.")
            exit(-1)
        elif run_name.split(".")[1] != header.dataset_prefix:
            logger.log_fatal("Neural run name dataset is not \"" + header.dataset_prefix + "\". Quit.")
            exit(-1)
        elif run_name.split(".")[3] != "resnet34mtl":
            logger.log_fatal("Unknown neural model \"" + run_name.split(".")[3] + "\". Quit.")
            exit(-1)

        header.config_neural["run_name"] = run_name
    else:
        header.config_neural["run_name"] = utility.generateRunName(header.config_neural["seed"], header.config_neural["type"], "resnet34mtl")

    if header.config_neural["run_name"] == "":
        logger.log_fatal("Missing neural run name. Quit.")
        exit(-1)

    header.config_neural["file_name_checkpoint"] = header.config_neural["run_name"] + header.checkpoint_postfix
    header.config_neural["file_name_checkpoint_best"] = header.config_neural["run_name"] + header.checkpoint_postfix_best

    return

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Seed.")
    arguments = parser.parse_args()

    initializeRunName(arguments.run_name)

    if arguments.seed is not None:
        header.config_neural["seed"] = arguments.seed

    logger.log_trace("Run name: \"" + header.config_neural["run_name"] + "\".")
    logger.log_trace("Seed: " + str(header.config_neural["seed"]) + ".")

    return

def test(model_neural, data_loader, device, batch_step):
    utility.loadCheckpoint(header.config_neural["file_name_checkpoint_best"], model_neural)

    accuracy_concept_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    tv_distance_epoch = 0
    tv_distances_epoch = []

    for _ in data_loader.dataset.config["attributes"]:
        tv_distances_epoch.append(0)

    model_neural.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels, _, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)

            for i in range(len(labels)):
                labels[i] = labels[i].to(device, non_blocking = True)

            (output, _) = model_neural(input)

            output = utility.applySoftmaxAttribute(output)
            accuracy_concept_batch = utility.computeConceptAccuracy(output, labels, device)

            accuracy_concept_epoch += accuracy_concept_batch

            for i in range(len(data_loader.dataset.config["attributes"])):
                tv_distance_batch = 0.5 * torch.sum(torch.abs(output[i] - labels[i]), dim = 1)
                tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy_concept": accuracy_concept_batch})
            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    accuracy_concept_epoch /= len(data_loader)

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distances_epoch[i] /= len(data_loader.dataset)

    tv_distance_epoch = sum(tv_distances_epoch) / len(tv_distances_epoch)

    wandb.log({"testing/epoch/accuracy_concept": accuracy_concept_epoch})
    wandb.log({"testing/epoch/tv_distance": tv_distance_epoch})

    wandb.summary["testing/epoch/accuracy_concept"] = accuracy_concept_epoch
    wandb.summary["testing/epoch/tv_distance"] = tv_distance_epoch

    logger.log_info("Testing mean TV distance: " + str(tv_distance_epoch) + ".")
    logger.log_info("Testing mean concept accuracy: " + str(accuracy_concept_epoch) + ".")

    return batch_step

def main():
    processArguments()

    utility.setSeed(header.config_neural["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_neural, mode = "disabled")

    dataset_transforms = utility.createTransforms(header.config_neural)
    dataset_test = dataset.NPCDataset(header.config_neural["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_neural["batch_size"], shuffle = False, num_workers = header.config_neural["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_neural = model.ResNet34MTL(dataset_test.config, device)
    model_neural = torch.nn.DataParallel(model_neural)
    model_neural = model_neural.to(device)

    test(model_neural, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
