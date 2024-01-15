import header
import json
import logger
import os
import PIL
import PIL.Image
import torch
import torchvision

class DatasetDecomposed(torch.utils.data.Dataset):
    def __init__(self, root, transform = None):
        self.class_to_idx = []
        self.class_to_idx_original = {}
        self.classes = []
        self.classes_original = []
        self.config = {}
        self.file_paths = []
        self.labels = []
        self.labels_original = []
        self.root = root
        self.transform = transform

        if not os.path.isdir(root):
            logger.log_error("Invalid dataset directory.")
            return

        dataset_test = torchvision.datasets.ImageFolder(header.config_baseline["dir_dataset_test"])
        self.classes_original = dataset_test.classes

        file_config_dataset = open(header.dataset_config_file_path, "r")
        self.config = json.load(file_config_dataset)
        file_config_dataset.close()

        for dataset in self.config["attributes"]:
            dataset_labels = dataset["labels"]
            dataset_labels.remove("")
            self.classes.append(dataset_labels)
            self.labels.append([])

        for classes_dataset in self.classes:
            class_to_idx_dataset = {}

            for (class_index, class_name) in enumerate(classes_dataset):
                class_to_idx_dataset[class_name] = class_index

            self.class_to_idx.append(class_to_idx_dataset)

        for (class_index, class_name) in enumerate(self.classes_original):
            self.class_to_idx_original[class_name] = class_index

        for class_name_original in self.classes_original:
            for file_name in os.listdir(os.path.join(root, class_name_original)):
                label_original = self.class_to_idx_original[class_name_original]

                self.file_paths.append(os.path.abspath(os.path.join(root, class_name_original, file_name)))
                self.labels_original.append(label_original)

                for (attribute_index, attribute) in enumerate(self.config["attributes"]):
                    attribute_name = attribute["name"]
                    class_name_decomposed = self.config["mappings"][class_name_original]["labels"][attribute_name]
                    label = self.class_to_idx[attribute_index][class_name_decomposed]
                    self.labels[attribute_index].append(label)

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
