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

def counterfactual_structural(spn_tree, device, batch, outputs_decomposed, labels_original):
    attribute_sizes = []
    induced_tree_candidates = []
    label_original = labels_original.cpu().tolist()[batch]
    leaf_nodes_original_ground_truth = set()
    moving_cost = 0

    for k in range(len(outputs_decomposed)):
        attribute_sizes.append(outputs_decomposed[k].shape[1])

    spn_tree.load(header.config_spn["file_path_spn"])
    spn_tree.gather_induced_trees()

    for i in range(len(spn_tree.induced_trees)):
        attributes = []
        leaf_nodes = set()
        leaf_nodes_original = set()
        filter = False

        for node in spn_tree.induced_trees[i][0]:
            if isinstance(node, spn.CategoricalLeafNode):
                if node in leaf_nodes:
                    logger.log_warn("Leaf node \"" + str(leaf_nodes) + "\" already encountered.")

                leaf_nodes.add(node)

        for leaf_node in leaf_nodes:
            if leaf_node.attribute_index < len(attribute_sizes):
                attributes.append((leaf_node.attribute_index, leaf_node.category_index))
            else:
                leaf_nodes_original.add(leaf_node)

                # Filter out induced SPNs with ground truth original class leaf nodes
                if leaf_node.category_index == label_original:
                    leaf_nodes_original_ground_truth.add(leaf_node)
                    filter = True

        if len(leaf_nodes_original) != 1:
            logger.log_warn("Obtained multiple original class leaf node categories.")

        if filter:
            logger.log_trace("Filtering out induced SPN with ground truth original class leaf node...")
            continue

        attributes.sort(key = lambda attribute: attribute[0])

        attribute_conditional_probability = 1
        spn_setting = []

        for attribute in attributes:
            index_attribute = attribute[0]
            index_category = attribute[1]
            attribute_conditional_probability *= outputs_decomposed[index_attribute][batch, index_category].item()
            spn_setting.append(index_category)

        spn_setting.append(-1)
        spn_setting = torch.Tensor(spn_setting).unsqueeze(0).to(device)
        spn_tree.set_leaf_nodes(spn_setting)
        attribute_marginal_probability = torch.exp(spn_tree.forward()).item()

        # Compute augmented weight
        spn_tree.induced_trees[i].append(spn_tree.induced_trees[i][1] * attribute_conditional_probability / attribute_marginal_probability)
        spn_tree.induced_trees[i].append(leaf_nodes_original)
        induced_tree_candidates.append(spn_tree.induced_trees[i])

    if len(leaf_nodes_original_ground_truth) != 1:
        logger.log_fatal("Invalid induced SPNs.")
        exit(-1)

    # Rank induced SPNs by augmented weight in descending order
    induced_tree_candidates.sort(key=lambda induced_tree: induced_tree[2], reverse = True)

    # Move edges
    for induced_tree_candidate in induced_tree_candidates:
        nodes = induced_tree_candidate[0]
        weight = induced_tree_candidate[1]
        leaf_node_original = list(induced_tree_candidate[3])[0]
        leaf_node_original_ground_truth = list(leaf_nodes_original_ground_truth)[0]

        moving_cost += weight

        if moving_cost > header.counterfactual_moving_epsilon:
            break

        # Locate original class leaf node parent in induced SPN
        leaf_node_original_parents = set(leaf_node_original.parents)
        leaf_node_original_parent = list(leaf_node_original_parents & nodes)

        if len(leaf_node_original_parent) != 1:
            logger.log_fatal("Failed to locate original class leaf node parent.")
            exit(-1)

        leaf_node_original_parent = leaf_node_original_parent[0]

        # Remove edge between original class leaf node and its parent in induced SPN
        leaf_node_original.parents.remove(leaf_node_original_parent)
        leaf_node_original_parent.children.remove(leaf_node_original)

        # Add edge between original class ground truth leaf node the parent of original class leaf node in induced SPN
        leaf_node_original_ground_truth.parents.append(leaf_node_original_parent)
        leaf_node_original_parent.children.append(leaf_node_original_ground_truth)

    return

def test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, spn_settings_marginal, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)
    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])

    accuracy_epoch_composed = 0
    accuracy_epoch_composed_counterfactual = 0
    instance_count_corrected = 0
    instance_count_incorrect = 0

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
        outputs_composed_counterfactual = []

        batch_size = labels_original.nelement()

        with torch.set_grad_enabled(False):
            (outputs_decomposed_original, _) = model_decomposed(input)
            outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed_original)

        (_, _, outputs_composed) = composition.Composition.spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)
        (_, predictions_composed) = torch.max(outputs_composed, 1)

        progress_bar_structure = tqdm.tqdm(total = batch_size, position = 1, leave = False)
        progress_bar_structure.set_description_str("[INFO]: Updating structure")

        for batch in range(batch_size):
            if predictions_composed[batch] == labels_original[batch]:
                output_composed_counterfactual = outputs_composed[batch]
            else:
                instance_count_incorrect += 1

                spn_joint_counterfactual = spn.SPN(device)
                spn_marginal_counterfactual = spn.SPN(device)

                counterfactual_structural(spn_joint_counterfactual, device, batch, outputs_decomposed, labels_original)
                counterfactual_structural(spn_marginal_counterfactual, device, batch, outputs_decomposed, labels_original)

                spn_joint_counterfactual.set_leaf_nodes(spn_settings_joint)
                spn_marginal_counterfactual.set_leaf_nodes(spn_settings_marginal)

                (_, _, output_composed_counterfactual) = composition.Composition.spn_single(batch, outputs_decomposed, spn_joint_counterfactual, spn_marginal_counterfactual, spn_output_rows, spn_output_cols, device)

            outputs_composed_counterfactual.append(output_composed_counterfactual)

            progress_bar_structure.n = batch
            progress_bar_structure.refresh()

        progress_bar_structure.close()

        outputs_composed_counterfactual = torch.stack(outputs_composed_counterfactual, 0)
        (_, predictions_composed_counterfactual) = torch.max(outputs_composed_counterfactual, 1)

        utility.saveCounterfactuals(input_file_paths, counterfactuals, data_loader.dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_decomposed, outputs_composed, outputs_composed_counterfactual)

        corrects_composed = torch.sum(predictions_composed == labels_original.data).item()
        corrects_composed_counterfactual = torch.sum(predictions_composed_counterfactual == labels_original.data).item()
        instance_count_corrected += corrects_composed_counterfactual - corrects_composed

        accuracy_epoch_composed += corrects_composed
        accuracy_epoch_composed_counterfactual += corrects_composed_counterfactual

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

    progress_bar.close()

    accuracy_epoch_composed /= len(data_loader.dataset)
    accuracy_epoch_composed_counterfactual /= len(data_loader.dataset)

    logger.log_info("Composed testing accuracy: " + str(accuracy_epoch_composed) + ".")
    logger.log_info("Composed counterfactual accuracy: " + str(accuracy_epoch_composed_counterfactual) + ".")
    logger.log_info("Correction rate: " + str(instance_count_corrected / instance_count_incorrect) + ".")

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

    test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, spn_settings_marginal, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
