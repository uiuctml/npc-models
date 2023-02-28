import json
import logger
import os
import PIL
import PIL.Image
import torch
import wandb

class DatasetGenerated(torch.utils.data.Dataset):
    def __init__(self, root, transform = None):
        self.classes = []
        self.config = {}
        self.file_paths = []
        self.labels = []
        self.root = root
        self.transform = transform

        if not os.path.isdir(root):
            logger.log_error("Invalid dataset directory.")
            return

        file_config_dataset = open(os.path.join(root, wandb.config.file_name_config_dataset), "r")
        self.config = json.load(file_config_dataset)
        file_config_dataset.close()

        for dataset in self.config["datasets"]:
            dataset_labels = dataset["labels"]
            dataset_labels.remove("")
            self.classes.append(dataset_labels)
            self.labels.append([])

        for file_name in os.listdir(root):
            if file_name == wandb.config.file_name_config_dataset or file_name == wandb.config.file_name_config_dataset_generation:
                continue

            file_name_split = file_name.split(wandb.config.dataset_delimiter_file_name)

            if len(file_name_split) < len(self.labels):
                logger.log_warn("Invalid data \"" + file_name + "\"")
                continue

            self.file_paths.append(os.path.abspath(os.path.join(root, file_name)))

            for i in range(0, len(self.labels)):
                label = self.classes[i].index(file_name_split[i])
                self.labels[i].append(label)

        return

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = PIL.Image.open(self.file_paths[index]).convert("RGB")
        labels = []

        if self.transform is not None:
            image = self.transform(image)

        for label in self.labels:
            labels.append(int(label[index]))

        return (image, torch.LongTensor(labels))

    def getDatasetClassCount(self, dataset_name):
        for dataset in self.config["datasets"]:
            if dataset["name"] == dataset_name:
                return len(dataset["labels"])
