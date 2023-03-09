#!/usr/bin/env python3

import dataset as dset
import header
import logger
import network
import torch
import torch.nn
import torchvision
import test
import utility
import wandb

def main():
    if not utility.initializeRunNameDecomposed() or header.run_name_decomposed == "":
        logger.log_error("Run name missing. Quit.")
        return

    wandb.init(config = header.config_decomposed, mode = "disabled")

    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.config_decomposed["model_input_height"], header.config_decomposed["model_input_width"])),
        torchvision.transforms.ToTensor(),
    ])
    dataset_original = torchvision.datasets.ImageFolder(header.config_baseline["dir_dataset_test"], dataset_transforms)
    dataset_test = dset.DatasetDecomposed(header.config_decomposed["dir_dataset_test"], dataset_original.classes, dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model = network.DecomposedNetworkA(config_dataset)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    test.testDecomposed(model, config_dataset, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
