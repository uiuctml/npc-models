#!/usr/bin/env python3

import argparse
import header
import logger
import spn
import torch
import utility
import wandb

def initializeRunName(run_name = "", optimizer = "cccp"):
    if run_name != "":
        if run_name.split(".")[1] != header.run_name_keyword_spn:
            logger.log_fatal("Invalid PC run name. Quit.")
            exit(-1)

        header.run_name_spn = run_name
    else:
        header.run_name_spn = utility.generateRunName(header.run_name_keyword_spn, optimizer, header.config_spn["seed"])

    if header.run_name_spn == "":
        logger.log_fatal("Missing PC run name. Quit.")
        exit(-1)

    header.config_spn["file_name_checkpoint"] = header.run_name_spn + ".tar"
    header.config_spn["file_name_checkpoint_best"] = header.run_name_spn + ".best.tar"
    header.config_spn["run_name"] = header.run_name_spn

    return

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Random seed.")
    arguments = parser.parse_args()

    if arguments.run_name == "":
        logger.log_info("Proceeding without run name.")
        header.run_name_spn == ""
        header.config_spn["file_name_checkpoint"] = ""
        header.config_spn["file_name_checkpoint_best"] = ""
        header.config_spn["run_name"] = ""
    else:
        initializeRunName(arguments.run_name)

    if arguments.seed is not None:
        header.config_spn["seed"] = arguments.seed

    logger.log_trace("Run name: \"" + header.run_name_spn + "\".")
    logger.log_trace("Random seed: " + str(header.config_spn["seed"]) + ".")

    return

def loadDataset(file_path_dataset, device):
    dataset = []

    with open(file_path_dataset, "r") as file_dataset:
        lines = file_dataset.readlines()

        for line in lines:
            line = line.strip()
            line_list = line.split(",")

            for i in range(len(line_list)):
                line_list[i] = int(line_list[i])

            dataset.append(line_list)

    dataset = torch.Tensor(dataset)
    dataset = dataset.to(device)

    return dataset

def test(spn_joint, settings_joint):
    log_likelihoods = spn_joint(settings_joint)
    log_likelihood = torch.log(torch.mean(torch.exp(log_likelihoods))).item()

    wandb.log({"testing/epoch/log_likelihood": log_likelihood})
    wandb.summary["testing/epoch/log_likelihood"] = log_likelihood
    logger.log_info("Testing log likelihood: " + str(log_likelihood) + ".")

    return

def main():
    processArguments()

    utility.setSeed(header.config_spn["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_spn, mode = "disabled")

    device = torch.device("cuda")
    dataset_test = loadDataset(header.config_spn["dir_dataset_test"], device)

    spn_joint = spn.SPN(device)

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depth: " + str(spn_joint.depth) + ".")

    if header.config_spn["run_name"] != "":
        utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])

    test(spn_joint, dataset_test)

    return

if __name__=="__main__":
    main()
