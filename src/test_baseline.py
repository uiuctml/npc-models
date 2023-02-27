#!/usr/bin/env python3

import config
import logger
import network
import sys
import torch
import torch.nn
import torchvision
import test
import wandb

def main():
    if len(sys.argv) > 1:
        config.run_name_baseline = sys.argv[1]
        config.config_baseline["file_name_checkpoint"] = config.run_name_baseline + ".tar"
        config.config_baseline["file_name_checkpoint_best"] = config.run_name_baseline + ".best.tar"
    else:
        logger.log_error("Run name missing. Quit.")
        return

    wandb.init(config = config.config_baseline, mode = "disabled")

    dataset = torchvision.datasets.ImageFolder(root = wandb.config.dir_dataset)
    device = torch.device("cuda")
    model = network.BaselineNetworkA(dataset)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    test.testBaseline(model, device, 1)

    return

if __name__ == "__main__":
    main()
