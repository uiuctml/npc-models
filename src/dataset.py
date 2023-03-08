import header
import json
import logger
import os
import PIL
import PIL.Image
import torch

class DatasetDecomposed(torch.utils.data.Dataset):
    def __init__(self, root, dataset_original, transform = None):
        self.classes = []
        self.classes_original = dataset_original.classes
        self.config = {}
        self.file_paths = []
        self.labels = []
        self.labels_original = []
        self.root = root
        self.transform = transform

        if not os.path.isdir(root):
            logger.log_error("Invalid dataset directory.")
            return

        file_config_dataset = open(os.path.join(root, header.config_decomposed["file_name_config_dataset"]), "r")
        self.config = json.load(file_config_dataset)
        file_config_dataset.close()

        for dataset in self.config["datasets"]:
            dataset_labels = dataset["labels"]
            dataset_labels.remove("")
            self.classes.append(dataset_labels)
            self.labels.append([])

        for file_name in os.listdir(root):
            if file_name == header.config_decomposed["file_name_config_dataset"] or file_name == header.config_decomposed["file_name_config_dataset_generation"]:
                continue

            file_name_split = file_name.split(header.config_decomposed["dataset_delimiter_file_name"])

            # A valid data should contain decomposed labels, an original label, and a file key
            if len(file_name_split) < len(self.labels) + 2:
                logger.log_warn("Invalid data \"" + file_name + "\".")
                continue

            class_original = file_name_split[-2]
            label_original = self.classes_original.index(class_original)

            self.file_paths.append(os.path.abspath(os.path.join(root, file_name)))
            self.labels_original.append(label_original)

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

        return (image, torch.LongTensor(labels), self.labels_original[index])
