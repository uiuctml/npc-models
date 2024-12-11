#!/usr/bin/env python3

import argument
import header
import json
import logger
import os
import spn
import torch
import type
import utility
import wandb

def loadDataset(dir_dataset, device):
    config_dataset = {}
    dataset = []

    with open(header.file_path_dataset_config, "r") as file_config_dataset:
        config_dataset = json.load(file_config_dataset)

    labels_attribute = utility.getLabelsAttribute(config_dataset)
    labels_original = utility.getLabelsOriginal(config_dataset)
    indices_attribute = utility.getIndicesFromLabelsAttribute(labels_attribute)
    indices_original = utility.getIndicesFromLabelsOriginal(labels_original)

    for _ in range(len(labels_attribute) + 1):
        dataset.append([])

    if "instance_wise" not in config_dataset or config_dataset["instance_wise"] == False:
        for class_name_original in labels_original:
            attributes = config_dataset["mappings"][class_name_original]["labels"]
            count_instances = len(os.listdir(os.path.join(dir_dataset, class_name_original)))

            for (i, attribute_name) in enumerate(attributes.keys()):
                category_names = attributes[attribute_name]
                count_categories = len(indices_attribute[attribute_name])
                binary_vector_category = torch.zeros(count_categories)

                if isinstance(category_names, list):
                    for category_name in category_names:
                        index_category = indices_attribute[attribute_name][category_name]
                        binary_vector_category[index_category] = 1
                else:
                    index_category = indices_attribute[attribute_name][category_names]
                    binary_vector_category[index_category] = 1

                for _ in range(count_instances):
                    dataset[i].append(binary_vector_category)

            count_original = len(indices_original)
            index_original = indices_original[class_name_original]
            binary_vector_original = torch.zeros(count_original)
            binary_vector_original[index_original] = 1

            for _ in range(count_instances):
                dataset[-1].append(binary_vector_original)
    else:
        for image_name in config_dataset["mappings"]:
            class_name_original = image_name.split('/')[0]
            attributes = config_dataset["mappings"][image_name]["labels"]

            for (i, attribute_name) in enumerate(attributes.keys()):
                category_names = attributes[attribute_name]
                count_categories = len(indices_attribute[attribute_name])
                binary_vector_category = torch.zeros(count_categories)

                if isinstance(category_names, list):
                    for category_name in category_names:
                        index_category = indices_attribute[attribute_name][category_name]
                        binary_vector_category[index_category] = 1
                else:
                    index_category = indices_attribute[attribute_name][category_names]
                    binary_vector_category[index_category] = 1

                dataset[i].append(binary_vector_category)

            count_original = len(indices_original)
            index_original = indices_original[class_name_original]
            binary_vector_original = torch.zeros(count_original)
            binary_vector_original[index_original] = 1

            dataset[-1].append(binary_vector_original)

    for i in range(len(dataset)):
        dataset[i] = torch.stack(dataset[i], dim = 0)
        dataset[i] = dataset[i].to(device)

    return dataset

def test(spn_joint, spn_marginal, settings_joint, settings_marginal, use_probability):
    log_likelihoods = spn_joint(settings_joint, False)

    if use_probability:
        log_likelihoods_marginal = spn_marginal(settings_marginal)
        log_likelihoods -= log_likelihoods_marginal

    log_likelihood = torch.mean(log_likelihoods).item()

    if use_probability:
        wandb.log({"testing/epoch/log_probability": log_likelihood})
        wandb.summary["testing/epoch/log_probability"] = log_likelihood
        logger.log_info("Testing log probability: " + str(log_likelihood) + ".")
    else:
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
    settings_marginal = torch.full((1, len(dataset_test)), -1).to(device)
    spn_joint = spn.SPN(device)
    spn_marginal = spn.SPN(device)
    use_probability = False

    if header.config_spn["optimizer"] == type.OptimizerSPN.pgd_generative.name:
        use_probability = True

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depth: " + str(spn_joint.depth) + ".")

    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    test(spn_joint, spn_marginal, dataset_test, settings_marginal, use_probability)

    return

if __name__=="__main__":
    main()
