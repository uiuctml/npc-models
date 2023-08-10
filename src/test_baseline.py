#!/usr/bin/env python3

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
    utility.processArgumentsTestBaseline()

    utility.setSeed(header.config_baseline["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    wandb.init(config = header.config_baseline, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_baseline)
    dataset_test = torchvision.datasets.ImageFolder(header.config_baseline["dir_dataset_test"], dataset_transforms)
    class_count = len(dataset_test.classes)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_baseline["data_loader_batch_size"], shuffle = False, num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model = network.createModelBaseline(class_count, device)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    if header.config_baseline["test_adversarial"]:
        test_adversarial.testAdversarialBaseline(model, data_loader_test, device, 1)
    else:
        test.testBaseline(model, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
