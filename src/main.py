#!/usr/bin/env python3

import header
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = numpy.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1000)  # pause a bit so that plots are updated

    return

def main():
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(root = header.dataset_dir_images, transform = data_transforms)
    dataset_split_lengths = [header.dataset_split_percentage_train, header.dataset_split_percentage_validation]
    (dataset_train, dataset_validation) = torch.utils.data.random_split(dataset, dataset_split_lengths)

    data_loader_training = torch.utils.data.DataLoader(dataset_train, batch_size = 4, shuffle = True, num_workers = 4)
    dataset_validation = torch.utils.data.DataLoader(dataset_train, batch_size = 4, shuffle = True, num_workers = 4)

    inputs, classes = next(iter(data_loader_training))

    out = torchvision.utils.make_grid(inputs)

    imshow(out, [dataset.classes[x] for x in classes])

    return

if __name__ == "__main__":
    main()
