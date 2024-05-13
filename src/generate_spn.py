#!/usr/bin/env python3

import header
import json
import os
import torch
import utility

def main():
    device = torch.device("cuda")

    file_config_dataset = open(header.file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    labels_attribute = utility.getLabelsAttribute(config_dataset)
    labels_attribute_indices = utility.getIndicesFromLabelsAttribute(labels_attribute)
    (_, label_probabilities) = utility.countAttributeJointProbabilities(config_dataset, device)

    cat_node_dict = {}
    lines_edges = "##EDGES##\n"
    lines_nodes = "##NODES##\n"
    node_sequence = 0
    node_sequence_root = node_sequence

    lines_nodes += str(node_sequence_root) + ",SUM\n"
    node_sequence += 1

    for (attribute_index, attribute) in enumerate(config_dataset["attributes"]):
        for category_index in labels_attribute_indices[attribute["name"]].values():
            line_cat_node = "CATNODEPRD," + str(attribute_index) + "," + str(category_index)
            cat_node_dict[line_cat_node] = node_sequence
            lines_nodes += str(node_sequence) + "," + line_cat_node + "\n"
            node_sequence += 1

    for probability in label_probabilities.values():
        category_indices = probability[0]
        frequency = probability[1]

        node_sequence_prd = node_sequence
        lines_nodes += str(node_sequence_prd) + ",PRD\n"
        node_sequence += 1

        lines_edges += str(node_sequence_root) + "," + str(node_sequence_prd) + "," + str(frequency) + "\n"

        for (attribute_index, category_index) in enumerate(category_indices):
            line_cat_node = "CATNODEPRD," + str(attribute_index) + "," + str(category_index)
            node_sequence_cat = cat_node_dict[line_cat_node]

            lines_edges += str(node_sequence_prd) + "," + str(node_sequence_cat) + "\n"

    lines = lines_nodes + lines_edges

    if not os.path.isdir(header.dir_output_spn):
        os.makedirs(header.dir_output_spn, exist_ok = True)

    with open(os.path.join(header.dir_output_spn, header.file_name_spn_manual), "w") as file_spn_manual:
        file_spn_manual.writelines(lines)

    return

if __name__ == "__main__":
    main()
