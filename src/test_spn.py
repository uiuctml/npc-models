#!/usr/bin/env python3

import header
import json
import logger
import spn
import torch
import utility

def main():
    file_dataset_config = open(header.file_path_dataset_config, "r")
    dataset_config = json.load(file_dataset_config)
    file_dataset_config.close()

    device = torch.device("cuda")
    labels_attribute = utility.getLabelsAttribute(dataset_config)
    labels_original = utility.getLabelsOriginal(dataset_config)
    indices_attribute = utility.getIndicesFromLabelsAttribute(labels_attribute)
    indices_original = utility.getIndicesFromLabelsOriginal(labels_original)
    spn_joint = spn.SPN()
    spn_marginal = spn.SPN()
    spn_settings_joint = []
    spn_settings_marginal = []

    for label_original in dataset_config["mappings"].keys():
        spn_setting_joint = []
        spn_setting_marginal = []

        for attribute_name in dataset_config["mappings"][label_original]["labels"].keys():
            label_attribute = dataset_config["mappings"][label_original]["labels"][attribute_name]
            index_attribute = indices_attribute[attribute_name][label_attribute]

            spn_setting_joint.append(index_attribute)
            spn_setting_marginal.append(index_attribute)

        index_original = indices_original[label_original]

        spn_setting_joint.append(index_original)
        spn_setting_marginal.append(-1)

        spn_settings_joint.append(spn_setting_joint)
        spn_settings_marginal.append(spn_setting_marginal)

    spn_settings_joint = torch.Tensor(spn_settings_joint)
    spn_settings_marginal = torch.Tensor(spn_settings_marginal)
    spn_settings_joint = spn_settings_joint.to(device)
    spn_settings_marginal = spn_settings_marginal.to(device)

    logger.log_info("Loading SPN from \"" + header.file_path_spn + "\"...")

    spn_joint.load(header.file_path_spn)
    spn_marginal.load(header.file_path_spn)

    logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
    logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
    logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
    logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
    logger.log_info("SPN depths: " + str(spn_joint.depth) + ".")

    logger.log_info("Setting SPN leaf nodes...")

    spn_joint.set_leaf_nodes(spn_settings_joint)
    spn_marginal.set_leaf_nodes(spn_settings_marginal)

    logger.log_info("Performing SPN forward pass...")

    log_likelihoods_joint = spn_joint.forward()
    log_likelihoods_marginal = spn_marginal.forward()

    logger.log_trace("Testing joint log likelihoods: " + str(log_likelihoods_joint) + ".")
    logger.log_trace("Testing marginal log likelihoods: " + str(log_likelihoods_marginal) + ".")

    probabilities = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)

    logger.log_info("Testing probabilities: " + str(probabilities) + ".")

    return

if __name__=="__main__":
    main()
