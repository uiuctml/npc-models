#!/usr/bin/env python3

import argparse
import header
import logger
import spn
import test_spn
import torch
import tqdm
import utility
import wandb

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type = int, default = 50, help = "Epochs.")
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Random seed.")
    arguments = parser.parse_args()

    header.config_spn["epochs"] = arguments.epochs
    header.config_spn["seed"] = arguments.seed

    test_spn.initializeRunName()

    logger.log_trace("Run name: \"" + header.run_name_spn + "\".")
    logger.log_trace("Epochs: " + str(header.config_spn["epochs"]) + ".")
    logger.log_trace("Random seed: " + str(header.config_spn["seed"]) + ".")

    return

def train(spn_joint, settings_joint, optimizer):
    log_likelihoods = spn_joint(settings_joint)

    spn_joint.backward()
    optimizer.step()

    log_likelihood = torch.log(torch.mean(torch.exp(log_likelihoods))).item()

    wandb.log({"training/epoch/log_likelihood": log_likelihood})

    return log_likelihood

def validate(spn_joint, settings_joint):
    log_likelihoods = spn_joint(settings_joint)
    log_likelihood = torch.log(torch.mean(torch.exp(log_likelihoods))).item()

    wandb.log({"validation/epoch/log_likelihood": log_likelihood})

    return log_likelihood

def main():
    processArguments()

    utility.setSeed(header.config_spn["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(project = header.project_name, name = header.run_name_spn, config = header.config_spn, mode = header.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + header.run_name_spn + "\".")

    device = torch.device("cuda")
    dataset_test = test_spn.loadDataset(header.config_spn["dir_dataset_test"], device)
    dataset_train = test_spn.loadDataset(header.config_spn["dir_dataset_train"], device)
    dataset_validation = test_spn.loadDataset(header.config_spn["dir_dataset_validation"], device)
    epoch = 1
    log_likelihood_validation_best = float("-inf")
    log_likelihood_train_epoch = 0
    log_likelihood_train_epoch_last = 0
    spn_joint = spn.SPN(device)
    spn_marginal = spn.SPN(device)
    optimizer = spn.CCCPGenerativeSPNOptimizer(spn_joint, spn_marginal, device)
    progress_bar = None
    learning_rate_scheduler = spn.LikelihoodSPNLearningRateScheduler(optimizer, header.config_spn["learning_rate_scheduler_factor"])

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])
    optimizer.set_weights_prior(spn_joint.get_weights())

    if header.config_spn["randomize_weights"]:
        logger.log_info("Randomizing SPN weights...")
        spn_joint.randomize_weights()
        spn_marginal.set_weights(spn_joint.get_weights())

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depth: " + str(spn_joint.depth) + ".")

    logger.log_info("Testing SPN...")
    test_spn.test(spn_joint, dataset_test)

    if epoch <= header.config_spn["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_spn["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_spn["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        log_likelihood_train_epoch_last = log_likelihood_train_epoch
        log_likelihood_train_epoch = train(spn_joint, dataset_train, optimizer)
        log_likelihood_validate_epoch = validate(spn_joint, dataset_validation)

        learning_rate_scheduler.step(log_likelihood_train_epoch)

        logger.log_info("Epoch validation log likelihood: " + str(log_likelihood_validate_epoch) + ".")

        if log_likelihood_validate_epoch > log_likelihood_validation_best or epoch == 1:
            log_likelihood_validation_best = log_likelihood_validate_epoch

            wandb.log({"validation/epoch/log_likelihood_best": log_likelihood_validation_best})

            utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch)

        utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch)

        if epoch > 1 and abs(log_likelihood_train_epoch - log_likelihood_train_epoch_last) < header.config_spn["stopping_criterion"]:
            logger.log_info("Stopping criterion reached.")
            break

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation log likelihood: " + str(log_likelihood_validation_best) + ".")
    wandb.summary["validation/epoch/log_likelihood_best"] = log_likelihood_validation_best

    wandb.log({"testing/epoch/step": 1})
    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    test_spn.test(spn_joint, dataset_test)

    return

if __name__=="__main__":
    main()
