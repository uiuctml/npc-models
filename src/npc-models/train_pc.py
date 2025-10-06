#!/usr/bin/env python3

import argparse
import header
import logger
import pc
import test_pc
import torch
import tqdm
import utility
import wandb

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type = int, default = None, help = "Epochs.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Random seed.")
    arguments = parser.parse_args()

    if arguments.epochs is not None:
        header.config_pc["epochs"] = arguments.epochs

    if arguments.seed is not None:
        header.config_pc["seed"] = arguments.seed

    test_pc.initializeRunName()

    logger.log_trace("Run name: \"" + header.config_pc["run_name"] + "\".")
    logger.log_trace("Epochs: " + str(header.config_pc["epochs"]) + ".")
    logger.log_trace("Random seed: " + str(header.config_pc["seed"]) + ".")

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

    utility.setSeed(header.config_pc["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(project = header.project_name, name = header.config_pc["run_name"], config = header.config_pc, mode = header.run_mode)

    utility.defineMetrics()

    logger.log_info("Started run \"" + header.config_pc["run_name"] + "\".")

    device = torch.device("cuda")
    dataset_test = test_pc.loadDataset(header.config_pc["file_path_dataset_test"], device)
    dataset_train = test_pc.loadDataset(header.config_pc["file_path_dataset_train"], device)
    dataset_validation = test_pc.loadDataset(header.config_pc["file_path_dataset_validation"], device)
    epoch = 1
    log_likelihood_validation_best = float("-inf")
    log_likelihood_train_epoch = 0
    log_likelihood_train_epoch_last = 0
    spn_joint = pc.SPN(device)
    spn_marginal = pc.SPN(device)
    optimizer = pc.CCCPSPNOptimizer(spn_joint, spn_marginal, device)
    progress_bar = None
    learning_rate_scheduler = pc.LikelihoodSPNLearningRateScheduler(optimizer, header.config_pc["learning_rate_scheduler_factor"])

    logger.log_info("Loading SPN from \"" + header.config_pc["file_path_pc"] + "\"...")

    spn_joint.load(header.config_pc["file_path_pc"])
    spn_marginal.load(header.config_pc["file_path_pc"])
    optimizer.set_weights_prior(spn_joint.get_weights())

    if header.config_pc["randomize_weights"]:
        logger.log_info("Randomizing SPN weights...")
        spn_joint.randomize_weights()
        spn_marginal.set_weights(spn_joint.get_weights())

    logger.log_trace("Total PC nodes: " + str(len(spn_joint.nodes)) + ".")
    logger.log_trace("Total PC sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
    logger.log_trace("Total PC product nodes: " + str(len(spn_joint.product_nodes)) + ".")
    logger.log_trace("Total PC leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
    logger.log_trace("SPN depth: " + str(spn_joint.depth) + ".")

    logger.log_info("Testing SPN...")
    test_pc.test(spn_joint, dataset_test)

    if epoch <= header.config_pc["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_pc["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_pc["epochs"]:
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

            utility.saveCheckpoint(header.config_pc["file_name_checkpoint_best"], spn_joint, True)

        utility.saveCheckpoint(header.config_pc["file_name_checkpoint"], spn_joint, True)

        if epoch > 1 and abs(log_likelihood_train_epoch - log_likelihood_train_epoch_last) < header.config_pc["stopping_criterion"]:
            logger.log_info("Stopping criterion reached.")
            break

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation log likelihood: " + str(log_likelihood_validation_best) + ".")
    wandb.summary["validation/epoch/log_likelihood_best"] = log_likelihood_validation_best

    wandb.log({"testing/epoch/step": 1})
    utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], spn_joint, True)
    test_pc.test(spn_joint, dataset_test)

    return

if __name__=="__main__":
    main()
