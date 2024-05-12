#!/usr/bin/env python3

import header
import json
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

    print(labels_attribute_indices)
    print(label_probabilities)
    print(len(label_probabilities))

    # TODO create cat leaf nodes for all individual attributes using labels_attribute_indices
    # TODO create product nodes using label_probabilities
    # TODO create edges using indices

    return

if __name__ == "__main__":
    main()
