#!/usr/bin/env python3

import argument
import header
import json
import logger
import spn
import torch
import utility

def generateSPNSettings(config_dataset, device):
    labels_attribute = utility.getLabelsAttribute(config_dataset)
    labels_original = utility.getLabelsOriginal(config_dataset)
    indices_attribute = utility.getIndicesFromLabelsAttribute(labels_attribute)
    indices_original = utility.getIndicesFromLabelsOriginal(labels_original)
    spn_settings = []

    for label_original in config_dataset["mappings"].keys():
        spn_setting = []

        for attribute_name in config_dataset["mappings"][label_original]["labels"].keys():
            label_attribute = config_dataset["mappings"][label_original]["labels"][attribute_name]
            index_attribute = indices_attribute[attribute_name][label_attribute]
            spn_setting.append(index_attribute)

        index_original = indices_original[label_original]
        spn_setting.append(index_original)
        spn_settings.append(spn_setting)

    spn_settings = torch.Tensor(spn_settings)
    spn_settings = spn_settings.to(device)

    return spn_settings

def main():
    argument.processArgumentsTestSPN()

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    file_dataset_config = open(header.file_path_dataset_config, "r")
    dataset_config = json.load(file_dataset_config)
    file_dataset_config.close()

    device = torch.device("cuda")
    spn_joint = spn.SPN()
    spn_marginal = spn.SPN()

    spn_settings_joint = generateSPNSettings(dataset_config, device)
    spn_settings_marginal = torch.clone(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])

    logger.log_info("Loading SPN leaf node settings...")

    spn_settings_joint = generateSPNSettings(dataset_config, device)
    spn_settings_marginal = torch.clone(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    logger.log_info("Setting SPN leaf nodes...")

    spn_joint.set_leaf_nodes(spn_settings_joint)
    spn_marginal.set_leaf_nodes(spn_settings_marginal)

    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depths: " + str(spn_joint.depth) + ".")
        logger.log_info("SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    log_likelihoods_joint = spn_joint.forward()
    log_likelihoods_marginal = spn_marginal.forward()

    logger.log_trace("Testing joint log likelihoods: " + str(log_likelihoods_joint) + ".")
    logger.log_trace("Testing marginal log likelihoods: " + str(log_likelihoods_marginal) + ".")

    probabilities = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)
    accuracy = torch.mean(probabilities).item()

    logger.log_info("Testing accuracy: " + str(accuracy) + ".")

    return

if __name__=="__main__":
    main()
