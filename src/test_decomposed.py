#!/usr/bin/env python3

import dataset as dset
import header
import logger
import network
import torch
import torch.nn
import test
import utility
import wandb

def main():
    if not utility.initializeRunNameDecomposed() or header.run_name_decomposed == "":
        logger.log_error("Run name missing. Quit.")
        return

    wandb.init(config = header.config_decomposed, mode = "disabled")

    dataset = dset.DatasetGenerated(root = header.config_decomposed["dir_dataset"])
    device = torch.device("cuda")
    model = network.DecomposedNetworkA(dataset)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    test.testDecomposed(model, dataset, device, 1)

    return

if __name__ == "__main__":
    main()
