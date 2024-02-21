#!/usr/bin/env python3

import argument
import header
import logger
import spn
import torch
import tqdm
import type
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

def test(spn_joint, settings):
    log_likelihoods = spn_joint(settings)
    log_likelihood_test = torch.mean(log_likelihoods).item()

    logger.log_info("Testing log likelihood: " + str(log_likelihood_test) + ".")

    return

def train(spn_joint, settings, optimizer):
    log_likelihoods = spn_joint(settings)

    spn_joint.backward()
    optimizer.step()

    return torch.mean(log_likelihoods).item()

def validate(spn_joint, settings):
    log_likelihoods = spn_joint(settings)

    return torch.mean(log_likelihoods).item()

def main():
    resume = argument.processArgumentsTrainSPN()

    utility.setSeed(header.config_spn["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(project = header.project_name, name = header.run_name_spn, config = header.config_spn, resume = resume, mode = header.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + header.run_name_spn + "\".")

    device = torch.device("cuda")
    dataset_test = loadDataset(header.config_spn["file_path_spn_dataset_test"], device)
    dataset_train = loadDataset(header.config_spn["file_path_spn_dataset_train"], device)
    dataset_validation = loadDataset(header.config_spn["file_path_spn_dataset_validation"], device)
    epoch = 1
    log_likelihood_best = float("-inf")
    log_likelihood_train = 0
    log_likelihood_train_last = 0
    spn_joint = spn.SPN(device)
    optimizer = None
    progress_bar = None

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])

    if header.config_spn["optimizer"] == type.OptimizerSPN.cccp_composed.name:
        optimizer = spn.CCCPComposedSPNOptimizer(spn_joint, None, device)
    elif header.config_spn["optimizer"] == type.OptimizerSPN.cccp_offline.name:
        optimizer = spn.CCCPOfflineSPNOptimizer(spn_joint, None, device)
    else:
        logger.log_fatal("Unknown SPN optimizer \"" + header.config_spn["optimizer"] + "\".")
        exit(-1)

    (log_likelihood_best, log_likelihood_train_last, epoch) = utility.loadCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], log_likelihood_best, log_likelihood_train_last, epoch)
    log_likelihood_train = log_likelihood_train_last

    if not resume and not header.config_spn["fine_tuning"]:
        spn_joint.randomize_weights()

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depths: " + str(spn_joint.depth) + ".")

    test(spn_joint, dataset_test)

    if epoch <= header.config_spn["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_spn["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_spn["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        log_likelihood_train_last = log_likelihood_train
        log_likelihood_train = train(spn_joint, dataset_train, optimizer)
        log_likelihood_validate = validate(spn_joint, dataset_validation)

        logger.log_info("Training log likelihood: " + str(log_likelihood_train) + ".")
        logger.log_info("Validation log likelihood: " + str(log_likelihood_validate) + ".")

        if log_likelihood_validate > log_likelihood_best:
            log_likelihood_best = log_likelihood_validate
            utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], log_likelihood_best, log_likelihood_train_last, epoch)

        utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], log_likelihood_best, log_likelihood_train_last, epoch)

        if epoch > 1 and log_likelihood_train - log_likelihood_train_last < header.config_spn["stopping_criterion"]:
            logger.log_info("Stopping criterion reached.")
            break

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    test(spn_joint, dataset_test)

    return

if __name__=="__main__":
    main()
