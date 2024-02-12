#!/usr/bin/env python3

import header
import logger
import spn
import torch

def load_dataset(file_path_dataset, device):
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

    return torch.mean(log_likelihoods).item()

def train(spn_joint, settings, optimizer):
    log_likelihoods = spn_joint(settings)

    spn_joint.backward()
    optimizer.step()

    return torch.mean(log_likelihoods).item()

def validate(spn_joint, settings):
    log_likelihoods = spn_joint(settings)

    return torch.mean(log_likelihoods).item()

def main():
    device = torch.device("cuda")
    dataset_test = load_dataset(header.file_path_spn_dataset_test, device)
    dataset_train = load_dataset(header.file_path_spn_dataset_train, device)
    dataset_validation = load_dataset(header.file_path_spn_dataset_validation, device)
    epoch_counter = 1
    epoch_first = True
    log_likelihood_best = float("-inf")
    log_likelihood_train = 0
    log_likelihood_train_last = 0
    spn_joint = spn.SPN(device)
    stopping_criterion = 1e-4
    optimizer = spn.CCCPOfflineSPNOptimizer(spn_joint, device)
    weights_best = None

    logger.log_info("Loading SPN from \"" + header.file_path_spn + "\"...")

    spn_joint.load(header.file_path_spn)
    weights_best = spn_joint.get_weights()

    logger.log_info("Number of nodes:", len(spn_joint.nodes))
    logger.log_info("Number of sum nodes:", len(spn_joint.sum_nodes))
    logger.log_info("Number of product nodes:", len(spn_joint.product_nodes))
    logger.log_info("Number of leaf nodes:", len(spn_joint.leaf_nodes))
    logger.log_info("SPN depths:", spn_joint.depth)

    log_likelihood_test = test(spn_joint, dataset_test)

    logger.log_info("Testing log likelihood:", round(log_likelihood_test, 4))

    while True:
        logger.log_info("Epoch:", epoch_counter)

        log_likelihood_train_last = log_likelihood_train

        log_likelihood_train = train(spn_joint, dataset_train, optimizer)
        log_likelihood_validate = validate(spn_joint, dataset_validation)

        logger.log_info("Training log likelihood:", round(log_likelihood_train, 4))
        logger.log_info("Validation log likelihood:", round(log_likelihood_validate, 4))

        if log_likelihood_validate > log_likelihood_best:
            log_likelihood_best = log_likelihood_validate
            weights_best = spn_joint.get_weights()

        if epoch_first:
            epoch_first = False
        elif log_likelihood_train - log_likelihood_train_last < stopping_criterion:
            break

        epoch_counter += 1

    spn_joint.set_weights(weights_best)

    log_likelihood_test = test(spn_joint, dataset_test)

    logger.log_info("Testing log likelihood:", round(log_likelihood_test, 4))

    return

if __name__=="__main__":
    main()
