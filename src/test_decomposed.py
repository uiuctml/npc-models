#!/usr/bin/env python3

import config
import dataset as dset
import logger
import network
import sys
import torch
import torch.nn
import test
import wandb

def main():
    if len(sys.argv) > 1:
        config.run_name_decomposed = sys.argv[1]
        config.config_decomposed["file_name_checkpoint"] = config.run_name_decomposed + ".tar"
        config.config_decomposed["file_name_checkpoint_best"] = config.run_name_decomposed + ".best.tar"
    else:
        logger.log_error("Run name missing. Quit.")
        return

    wandb.init(config = config.config_decomposed, mode = "disabled")

    dataset = dset.DatasetGenerated(root = wandb.config.dir_dataset)
    device = torch.device("cuda")
    model = network.DecomposedNetworkA(dataset)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    test.testDecomposed(model, dataset, None, device, 1)

    return

if __name__ == "__main__":
    main()
