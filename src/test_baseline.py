#!/usr/bin/env python3

import header
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
        header.run_name_baseline = sys.argv[1]
        header.config_baseline["file_name_checkpoint"] = header.run_name_baseline + ".tar"
        header.config_baseline["file_name_checkpoint_best"] = header.run_name_baseline + ".best.tar"
    else:
        logger.log_error("Run name missing. Quit.")
        return

    wandb.init(config = header.config_baseline, mode = "disabled")

    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.config_baseline["model_input_height"], header.config_baseline["model_input_width"])),
        torchvision.transforms.ToTensor(),
    ])
    dataset_test = torchvision.datasets.ImageFolder(root = header.config_baseline["dir_dataset_test"], transform = dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_baseline["data_loader_batch_size"], shuffle = False, num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model = network.BaselineNetworkA(dataset_test)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    test.testBaseline(model, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
