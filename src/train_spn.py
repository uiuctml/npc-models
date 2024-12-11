#!/usr/bin/env python3

import argument
import header
import logger
import spn
import test_spn
import torch
import tqdm
import type
import utility
import wandb

def train(spn_joint, spn_marginal, settings_joint, settings_marginal, optimizer, use_probability):
    log_likelihoods = spn_joint(settings_joint, False)

    spn_joint.backward()

    if use_probability:
        log_likelihoods_marginal = spn_marginal(settings_marginal)
        spn_marginal.backward()
        log_likelihoods -= log_likelihoods_marginal

    optimizer.step()

    log_likelihood = torch.mean(log_likelihoods).item()

    if use_probability:
        wandb.log({"training/epoch/log_probability": log_likelihood})
    else:
        wandb.log({"training/epoch/log_likelihood": log_likelihood})

    return log_likelihood

def validate(spn_joint, spn_marginal, settings_joint, settings_marginal, use_probability):
    log_likelihoods = spn_joint(settings_joint, False)

    if use_probability:
        log_likelihoods_marginal = spn_marginal(settings_marginal)
        log_likelihoods -= log_likelihoods_marginal

    log_likelihood = torch.mean(log_likelihoods).item()

    if use_probability:
        wandb.log({"validation/epoch/log_probability": log_likelihood})
    else:
        wandb.log({"validation/epoch/log_likelihood": log_likelihood})

    return log_likelihood

def main():
    resume = argument.processArgumentsTrainSPN()

    utility.setSeed(header.config_spn["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(project = header.project_name, name = header.run_name_spn, config = header.config_spn, resume = resume, mode = header.run_mode)

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
    normalize = False
    use_probability = False
    settings_marginal = torch.full((1, len(dataset_test)), -1).to(device)
    spn_joint = spn.SPN(device)
    spn_marginal = spn.SPN(device)
    optimizer = None
    progress_bar = None

    if header.config_spn["optimizer"] == type.OptimizerSPN.cccp_generative.name:
        optimizer = spn.CCCPGenerativeSPNOptimizer(spn_joint, spn_marginal, device)
    elif header.config_spn["optimizer"] == type.OptimizerSPN.pgd_generative.name:
        optimizer = spn.PGDGenerativeSPNOptimizer(spn_joint, spn_marginal, device, header.config_spn["optimizer_learning_rate"], header.config_spn["optimizer_prior_factor"], header.config_spn["epsilon_projection"])
        normalize = True
        use_probability = True
    else:
        logger.log_fatal("Unknown or unsupported SPN optimizer \"" + header.config_spn["optimizer"] + "\".")
        exit(-1)

    learning_rate_scheduler = spn.LikelihoodSPNLearningRateScheduler(optimizer, header.config_spn["learning_rate_scheduler_factor"])

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])
    optimizer.set_weights_prior(spn_joint.get_weights())

    (log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch) = utility.loadCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch)
    log_likelihood_train_epoch = log_likelihood_train_epoch_last

    if header.config_spn["fine_tuning"]:
        utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["model_pretrained_weights"])
        utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["model_pretrained_weights"])
    elif header.config_spn["randomize_weights"]:
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
    test_spn.test(spn_joint, spn_marginal, dataset_test, settings_marginal, use_probability)

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
        log_likelihood_train_epoch = train(spn_joint, spn_marginal, dataset_train, settings_marginal, optimizer, use_probability)
        log_likelihood_validate_epoch = validate(spn_joint, spn_marginal, dataset_validation, settings_marginal, use_probability)

        learning_rate_scheduler.step(log_likelihood_train_epoch)

        if use_probability:
            logger.log_info("Epoch validation log probability: " + str(log_likelihood_validate_epoch) + ".")
        else:
            logger.log_info("Epoch validation log likelihood: " + str(log_likelihood_validate_epoch) + ".")

        if log_likelihood_validate_epoch > log_likelihood_validation_best or epoch == 1:
            log_likelihood_validation_best = log_likelihood_validate_epoch

            if use_probability:
                wandb.log({"validation/epoch/log_probability_best": log_likelihood_validation_best})
            else:
                wandb.log({"validation/epoch/log_likelihood_best": log_likelihood_validation_best})

            utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch)

        utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch)

        if epoch > 1 and abs(log_likelihood_train_epoch - log_likelihood_train_epoch_last) < header.config_spn["stopping_criterion"]:
            logger.log_info("Stopping criterion reached.")
            break

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    if use_probability:
        logger.log_info("Best validation log probability: " + str(log_likelihood_validation_best) + ".")
        wandb.summary["validation/epoch/log_probability_best"] = log_likelihood_validation_best
    else:
        logger.log_info("Best validation log likelihood: " + str(log_likelihood_validation_best) + ".")
        wandb.summary["validation/epoch/log_likelihood_best"] = log_likelihood_validation_best

    if normalize:
        wandb.run.resumed = True

        logger.log_info("Normalizing SPN weights...")

        utility.loadCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], 0, 0, 0)
        spn_marginal(settings_marginal)
        spn_marginal.normalize_weights(header.config_spn["epsilon_smoothing"])
        utility.saveCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch)

        (log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch) = utility.loadCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch)
        spn_marginal(settings_marginal)
        spn_marginal.normalize_weights(header.config_spn["epsilon_smoothing"])
        utility.saveCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], log_likelihood_validation_best, log_likelihood_train_epoch_last, epoch)

        wandb.run.resumed = False

    wandb.log({"testing/epoch/step": 1})
    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    test_spn.test(spn_joint, spn_marginal, dataset_test, settings_marginal, use_probability)

    return

if __name__=="__main__":
    main()
