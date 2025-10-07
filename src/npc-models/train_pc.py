#!/usr/bin/env python3

import argparse
import header
import logger
import math
import pc
import test_pc
import torch
import tqdm
import utility
import wandb

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type = int, default = None, help = "Epochs.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Seed.")
    arguments = parser.parse_args()

    if arguments.epochs is not None:
        header.config_pc["epochs"] = arguments.epochs

    if arguments.seed is not None:
        header.config_pc["seed"] = arguments.seed

    test_pc.initializeRunName()

    logger.log_trace("Run name: \"" + header.config_pc["run_name"] + "\".")
    logger.log_trace("Epochs: " + str(header.config_pc["epochs"]) + ".")
    logger.log_trace("Seed: " + str(header.config_pc["seed"]) + ".")

    return

def train(pc_joint, settings_joint, optimizer):
    log_likelihoods = pc_joint(settings_joint)

    pc_joint.backward()
    optimizer.step()

    log_likelihood = torch.log(torch.mean(torch.exp(log_likelihoods))).item()

    wandb.log({"training/epoch/log_likelihood": log_likelihood})

    return log_likelihood

def validate(pc_joint, settings_joint):
    log_likelihoods = pc_joint(settings_joint)
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
    pc_joint = pc.ProbabilisticCircuit(device)
    pc_marginal = pc.ProbabilisticCircuit(device)
    optimizer = pc.CCCPPCOptimizer(pc_joint, pc_marginal, device)
    learning_rate_scheduler = pc.LikelihoodPCLearningRateScheduler(optimizer, header.config_pc["learning_rate_scheduler_factor"])

    logger.log_info("Loading PC \"" + header.config_pc["file_path_pc"] + "\"...")

    pc_joint.load(header.config_pc["file_path_pc"])
    pc_marginal.load(header.config_pc["file_path_pc"])
    optimizer.set_weights_prior(pc_joint.get_weights())

    if header.config_pc["randomize_weights"]:
        logger.log_info("Randomizing PC weights...")
        pc_joint.randomize_weights()
        pc_marginal.set_weights(pc_joint.get_weights())

    logger.log_trace("Total PC nodes: " + str(len(pc_joint.nodes)) + ".")
    logger.log_trace("Total PC sum nodes: " + str(len(pc_joint.sum_nodes)) + ".")
    logger.log_trace("Total PC product nodes: " + str(len(pc_joint.product_nodes)) + ".")
    logger.log_trace("Total PC leaf nodes: " + str(len(pc_joint.leaf_nodes)) + ".")
    logger.log_trace("PC depth: " + str(pc_joint.depth) + ".")

    logger.log_info("Testing PC...")
    test_pc.test(pc_joint, dataset_test)

    progress_bar = tqdm.tqdm(total = header.config_pc["epochs"], position = 0)

    progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_pc["epochs"]:
        progress_bar.n = epoch
        progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        log_likelihood_train_epoch_last = log_likelihood_train_epoch
        log_likelihood_train_epoch = train(pc_joint, dataset_train, optimizer)
        log_likelihood_validate_epoch = validate(pc_joint, dataset_validation)

        learning_rate_scheduler.step(log_likelihood_train_epoch)

        logger.log_info("Validation log mean likelihood: " + str(log_likelihood_validate_epoch) + ".")
        logger.log_info("Validation mean likelihood: " + str(math.exp(log_likelihood_validate_epoch)) + ".")

        if log_likelihood_validate_epoch > log_likelihood_validation_best or epoch == 1:
            log_likelihood_validation_best = log_likelihood_validate_epoch

            wandb.log({"validation/epoch/log_likelihood_best": log_likelihood_validation_best})

            utility.saveCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_joint, True)

        utility.saveCheckpoint(header.config_pc["file_name_checkpoint"], pc_joint, True)

        if epoch > 1 and abs(log_likelihood_train_epoch - log_likelihood_train_epoch_last) < header.config_pc["stopping_criterion"]:
            logger.log_info("Stopping criterion reached.")
            break

        epoch += 1

    progress_bar.close()

    logger.log_info("Best validation log mean likelihood: " + str(log_likelihood_validation_best) + ".")
    logger.log_info("Best validation mean likelihood: " + str(math.exp(log_likelihood_validation_best)) + ".")
    wandb.summary["validation/epoch/log_likelihood_best"] = log_likelihood_validation_best

    wandb.log({"testing/epoch/step": 1})
    utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_joint, True)
    test_pc.test(pc_joint, dataset_test)

    return

if __name__=="__main__":
    main()
