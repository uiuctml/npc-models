#!/usr/bin/env python3

import header
import logger
import spn
import torch

def main():
    device = torch.device("cuda")
    spn_joint = spn.SPN()
    spn_marginal = spn.SPN()
    # GTSRB ground truth attribute and original labels
    spn_settings_gtsrb_joint = torch.Tensor([
        [2, 0, 0, 9, 15, 2, 2, 0],
        [2, 0, 0, 9, 15, 2, 3, 1],
        [2, 0, 0, 9, 15, 2, 4, 2],
        [2, 0, 0, 9, 15, 2, 5, 3],
        [2, 0, 0, 9, 15, 2, 6, 4],
        [2, 0, 0, 9, 15, 2, 7, 5],
        [2, 1, 0, 9, 15, 2, 7, 6],
        [2, 0, 0, 9, 15, 2, 0, 7],
        [2, 0, 0, 9, 15, 2, 1, 8],
        [2, 0, 0, 9,  4, 1, 8, 9],
        [2, 0, 0, 9,  3, 1, 8, 10],
        [2, 0, 3, 6, 15, 0, 8, 11],
        [2, 2, 1, 9, 15, 3, 8, 12],
        [2, 0, 3, 9, 15, 3, 8, 13],
        [1, 2, 2, 9, 12, 1, 8, 14],
        [2, 0, 0, 9, 15, 3, 8, 15],
        [2, 0, 0, 9, 14, 1, 8, 16],
        [1, 1, 0, 9, 15, 3, 8, 17],
        [2, 0, 3, 9,  6, 1, 8, 18],
        [2, 0, 3, 3, 15, 0, 8, 19],
        [2, 0, 3, 4, 15, 0, 8, 20],
        [2, 0, 3, 0, 15, 0, 8, 21],
        [2, 0, 3, 9,  1, 1, 8, 22],
        [2, 0, 3, 9,  2, 1, 8, 23],
        [2, 0, 3, 9, 10, 1, 8, 24],
        [2, 0, 3, 9, 11, 1, 8, 25],
        [2, 0, 3, 9, 13, 1, 8, 26],
        [2, 0, 3, 9,  8, 1, 8, 27],
        [2, 0, 3, 9,  9, 1, 8, 28],
        [2, 0, 3, 9,  0, 1, 8, 29],
        [2, 0, 3, 9,  7, 1, 8, 30],
        [2, 0, 3, 9,  5, 1, 8, 31],
        [2, 1, 0, 9, 15, 3, 8, 32],
        [0, 2, 0, 4, 15, 0, 8, 33],
        [0, 2, 0, 3, 15, 0, 8, 34],
        [0, 2, 0, 6, 15, 0, 8, 35],
        [0, 2, 0, 8, 15, 0, 8, 36],
        [0, 2, 0, 7, 15, 0, 8, 37],
        [0, 2, 0, 2, 15, 0, 8, 38],
        [0, 2, 0, 1, 15, 0, 8, 39],
        [0, 2, 0, 5, 15, 0, 8, 40],
        [2, 1, 0, 9,  4, 1, 8, 41],
        [2, 1, 0, 9,  3, 1, 8, 42]
    ])
    # GTSRB ground truth attribute labels with original labels marginalized
    spn_settings_gtsrb_marginal = torch.Tensor([
        [2, 0, 0, 9, 15, 2, 2, -1],
        [2, 0, 0, 9, 15, 2, 3, -1],
        [2, 0, 0, 9, 15, 2, 4, -1],
        [2, 0, 0, 9, 15, 2, 5, -1],
        [2, 0, 0, 9, 15, 2, 6, -1],
        [2, 0, 0, 9, 15, 2, 7, -1],
        [2, 1, 0, 9, 15, 2, 7, -1],
        [2, 0, 0, 9, 15, 2, 0, -1],
        [2, 0, 0, 9, 15, 2, 1, -1],
        [2, 0, 0, 9,  4, 1, 8, -1],
        [2, 0, 0, 9,  3, 1, 8, -1],
        [2, 0, 3, 6, 15, 0, 8, -1],
        [2, 2, 1, 9, 15, 3, 8, -1],
        [2, 0, 3, 9, 15, 3, 8, -1],
        [1, 2, 2, 9, 12, 1, 8, -1],
        [2, 0, 0, 9, 15, 3, 8, -1],
        [2, 0, 0, 9, 14, 1, 8, -1],
        [1, 1, 0, 9, 15, 3, 8, -1],
        [2, 0, 3, 9,  6, 1, 8, -1],
        [2, 0, 3, 3, 15, 0, 8, -1],
        [2, 0, 3, 4, 15, 0, 8, -1],
        [2, 0, 3, 0, 15, 0, 8, -1],
        [2, 0, 3, 9,  1, 1, 8, -1],
        [2, 0, 3, 9,  2, 1, 8, -1],
        [2, 0, 3, 9, 10, 1, 8, -1],
        [2, 0, 3, 9, 11, 1, 8, -1],
        [2, 0, 3, 9, 13, 1, 8, -1],
        [2, 0, 3, 9,  8, 1, 8, -1],
        [2, 0, 3, 9,  9, 1, 8, -1],
        [2, 0, 3, 9,  0, 1, 8, -1],
        [2, 0, 3, 9,  7, 1, 8, -1],
        [2, 0, 3, 9,  5, 1, 8, -1],
        [2, 1, 0, 9, 15, 3, 8, -1],
        [0, 2, 0, 4, 15, 0, 8, -1],
        [0, 2, 0, 3, 15, 0, 8, -1],
        [0, 2, 0, 6, 15, 0, 8, -1],
        [0, 2, 0, 8, 15, 0, 8, -1],
        [0, 2, 0, 7, 15, 0, 8, -1],
        [0, 2, 0, 2, 15, 0, 8, -1],
        [0, 2, 0, 1, 15, 0, 8, -1],
        [0, 2, 0, 5, 15, 0, 8, -1],
        [2, 1, 0, 9,  4, 1, 8, -1],
        [2, 1, 0, 9,  3, 1, 8, -1]
    ])

    logger.log_info("Loading SPN from \"" + header.file_path_spn + "\"...")

    spn_joint.load(header.file_path_spn)
    spn_marginal.load(header.file_path_spn)

    logger.log_info("Number of nodes:", len(spn_joint.nodes))
    logger.log_info("Number of sum nodes:", len(spn_joint.sum_nodes))
    logger.log_info("Number of product nodes:", len(spn_joint.product_nodes))
    logger.log_info("Number of leaf nodes:", len(spn_joint.leaf_nodes))
    logger.log_info("SPN depths:", spn_joint.depth)

    logger.log_info("Setting SPN...")

    spn_settings_gtsrb_joint = spn_settings_gtsrb_joint.to(device)
    spn_settings_gtsrb_marginal = spn_settings_gtsrb_marginal.to(device)

    spn_joint.set(spn_settings_gtsrb_joint)
    spn_marginal.set(spn_settings_gtsrb_joint)

    logger.log_info("Performing SPN forward pass...")

    log_likelihoods_joint = spn_joint.forward()
    log_likelihoods_marginal = spn_marginal.forward()

    logger.log_trace("Forward joint log likelihoods:", log_likelihoods_joint)
    logger.log_trace("Forward marginal log likelihoods:", log_likelihoods_marginal)

    probabilities = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)

    logger.log_info("Forward probabilities:", probabilities)

    logger.log_info("Performing SPN backward pass...")

    spn_joint.backward()

    return

if __name__=="__main__":
    main()
