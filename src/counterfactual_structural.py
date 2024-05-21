#!/usr/bin/env python3

import argument
import composition
import dataset
import header
import json
import logger
import model
import os
import spn
import torch
import tqdm
import utility

def counterfactual_structural(config_dataset, device):
    spn_joint = spn.SPN(device)
    spn_marginal = spn.SPN(device)

    logger.log_info("Loading counterfactual SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])

    logger.log_info("Loading counterfactual SPN leaf node settings...")

    spn_settings_joint = utility.generateSPNSettings(config_dataset, device)
    spn_settings_marginal = torch.clone(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    logger.log_info("Setting counterfactual SPN leaf nodes...")

    spn_joint.set_leaf_nodes(spn_settings_joint)
    spn_marginal.set_leaf_nodes(spn_settings_marginal)

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("Counterfactual SPN depths: " + str(spn_joint.depth) + ".")
        logger.log_info("Counterfactual SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    spn_joint.gather_induced_trees()

    return (spn_joint, spn_marginal)

def save(input_file_paths, counterfactuals, dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_decomposed_counterfactual, outputs_composed, outputs_composed_counterfactual):
    batch_size = labels_original.nelement()
    config_dataset = dataset.config

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        counterfactuals[input_file_path] = {}
        counterfactuals[input_file_path]["counterfactual"] = {}
        counterfactuals[input_file_path]["ground_truth"] = {}
        counterfactuals[input_file_path]["prediction"] = {}

        for (attribute_index, attribute) in enumerate(config_dataset["attributes"]):
            attribute_name = attribute["name"]

            (outputs_decomposed_counterfactual_mpe_probability, outputs_decomposed_counterfactual_mpe_label) = torch.max(outputs_decomposed_counterfactual[attribute_index][batch_index], 0)
            (outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label) = torch.max(outputs_decomposed[attribute_index][batch_index], 0)

            counterfactuals[input_file_path]["counterfactual"][attribute_name] = (dataset.classes[attribute_name][outputs_decomposed_counterfactual_mpe_label], outputs_decomposed_counterfactual_mpe_probability.item())
            counterfactuals[input_file_path]["ground_truth"][attribute_name] = dataset.classes[attribute_name][labels_decomposed[batch_index][attribute_index]]
            counterfactuals[input_file_path]["prediction"][attribute_name] = (dataset.classes[attribute_name][outputs_decomposed_mpe_label], outputs_decomposed_mpe_probability.item())

        (outputs_composed_counterfactual_mpe_probability, outputs_composed_counterfactual_mpe_label) = torch.max(outputs_composed_counterfactual[batch_index], 0)
        (outputs_composed_mpe_probability, outputs_composed_mpe_label) = torch.max(outputs_composed[batch_index], 0)

        counterfactuals[input_file_path]["counterfactual"]["original"] = (dataset.classes_original[outputs_composed_counterfactual_mpe_label], outputs_composed_counterfactual_mpe_probability.item())
        counterfactuals[input_file_path]["ground_truth"]["original"] = dataset.classes_original[labels_original[batch_index]]
        counterfactuals[input_file_path]["prediction"]["original"] = (dataset.classes_original[outputs_composed_mpe_label], outputs_composed_mpe_probability.item())

    return

def test(model_decomposed, spn_joint, spn_marginal, spn_joint_counterfactual, spn_marginal_counterfactual, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)
    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])

    accuracy_epoch_composed = 0
    accuracy_epoch_composed_counterfactual = 0

    config_dataset = data_loader.dataset.config
    counterfactuals = {}
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    spn_output_rows = len(data_loader.dataset.classes_original)
    spn_output_cols = 1

    for attribute in config_dataset["attributes"]:
        spn_output_cols *= len(attribute["labels"])

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Counterfactual progress")

    for (batch_index, (input, labels_decomposed, labels_original, input_file_paths)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels_decomposed = labels_decomposed.to(device, non_blocking = True)
        labels_original = labels_original.to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            (outputs_decomposed_original, _) = model_decomposed(input)
            outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed_original)

        (_, _, outputs_composed) = composition.Composition.spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)
        (_, _, outputs_composed_counterfactual) = composition.Composition.spn(outputs_decomposed, spn_joint_counterfactual, spn_marginal_counterfactual, spn_output_rows, spn_output_cols, device)

        save(input_file_paths, counterfactuals, data_loader.dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_decomposed, outputs_composed, outputs_composed_counterfactual)

        (_, predictions_composed) = torch.max(outputs_composed, 1)
        (_, predictions_composed_counterfactual) = torch.max(outputs_composed_counterfactual, 1)

        corrects_composed = torch.sum(predictions_composed == labels_original.data).item()
        corrects_composed_counterfactual = torch.sum(predictions_composed_counterfactual == labels_original.data).item()

        accuracy_epoch_composed += corrects_composed
        accuracy_epoch_composed_counterfactual += corrects_composed_counterfactual

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

    progress_bar.close()

    accuracy_epoch_composed /= len(data_loader.dataset)
    accuracy_epoch_composed_counterfactual /= len(data_loader.dataset)

    logger.log_info("Composed testing accuracy: " + str(accuracy_epoch_composed) + ".")
    logger.log_info("Composed counterfactual accuracy: " + str(accuracy_epoch_composed_counterfactual) + ".")

    if not os.path.isdir(header.dir_output_counterfactual):
        os.makedirs(header.dir_output_counterfactual, exist_ok = True)

    with open(os.path.join(header.dir_output_counterfactual, header.file_name_counterfactual), "w") as file_counterfactual:
        json.dump(counterfactuals, file_counterfactual, indent = 4)

    return

def main():
    argument.processArgumentsTestComposed()

    header.run_name_baseline = header.config_decomposed["run_name"]
    header.config_baseline["dir_dataset_test"] = header.config_decomposed["dir_dataset_test"]
    header.config_baseline["file_name_checkpoint"] = header.run_name_decomposed + ".tar"
    header.config_baseline["file_name_checkpoint_best"] = header.run_name_decomposed + ".best.tar"
    header.config_baseline["run_name"] = header.run_name_decomposed

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    spn_joint = spn.SPN(device)
    spn_marginal = spn.SPN(device)

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])

    logger.log_info("Loading SPN leaf node settings...")

    spn_settings_joint = utility.generateSPNSettings(config_dataset, device)
    spn_settings_marginal = torch.clone(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    logger.log_info("Setting SPN leaf nodes...")

    spn_joint.set_leaf_nodes(spn_settings_joint)
    spn_marginal.set_leaf_nodes(spn_settings_marginal)

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depths: " + str(spn_joint.depth) + ".")
        logger.log_info("SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    (spn_joint_counterfactual, spn_marginal_counterfactual) = counterfactual_structural(config_dataset, device)
    test(model_decomposed, spn_joint, spn_marginal, spn_joint_counterfactual, spn_marginal_counterfactual, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
