#!/usr/bin/env python3

import argument
import header
import logger
import spn
import torch
import utility
import wandb

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
    argument.processArgumentsTestSPN()

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
