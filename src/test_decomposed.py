#!/usr/bin/env python3

import dataset as dset
import header
import network
import torch
import torch.nn
import torchvision
import test
import test_adversarial
import utility
import wandb

def main():
    utility.processArgumentsTestDecomposed()

    utility.setSeed(header.config_decomposed["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    wandb.init(config = header.config_decomposed, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_original = torchvision.datasets.ImageFolder(header.config_baseline["dir_dataset_test"], dataset_transforms)
    dataset_test = dset.DatasetDecomposed(header.config_decomposed["dir_dataset_test"], dataset_original.classes, dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model = network.createModelDecomposed(config_dataset, device)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    if header.config_decomposed["test_adversarial"]:
        test_adversarial.testAdversarialDecomposed(model, config_dataset, data_loader_test, device, 1)
    else:
        test.testDecomposed(model, config_dataset, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
