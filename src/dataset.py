import config
import json
import logger
import os
import PIL
import PIL.Image
import torch

class DatasetGenerated(torch.utils.data.Dataset):
    def constructOriginalKeyLabelMap(self):
        self.classes_original = []
        self.key_label_map = {}

        dir_dataset_original = "../../mapillary-dataset/images/sliced/original"
        dir_dataset_original_list = os.listdir(dir_dataset_original)

        for dir_dataset_original_label in dir_dataset_original_list:
            dir_dataset_original_label_list = os.listdir(os.path.join(dir_dataset_original, dir_dataset_original_label))

            for dir_dataset_original_image in dir_dataset_original_label_list:
                key = dir_dataset_original_image.split(".")[0]
                self.key_label_map[key] = dir_dataset_original_label

        file_config_dataset_generation = open(os.path.join(config.config_decomposed["dir_dataset"], config.config_decomposed["file_name_config_dataset_generation"]), "r")
        config_dataset_generation = json.load(file_config_dataset_generation)
        file_config_dataset_generation.close()

        for class_name in config_dataset_generation.keys():
            self.classes_original.append(class_name)

        return

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

        self.constructOriginalKeyLabelMap()

        file_config_dataset = open(os.path.join(root, config.config_decomposed["file_name_config_dataset"]), "r")
        self.config = json.load(file_config_dataset)
        file_config_dataset.close()

        for dataset in self.config["datasets"]:
            dataset_labels = dataset["labels"]
            dataset_labels.remove("")
            self.classes.append(dataset_labels)
            self.labels.append([])

        for file_name in os.listdir(root):
            if file_name == config.config_decomposed["file_name_config_dataset"] or file_name == config.config_decomposed["file_name_config_dataset_generation"]:
                continue

            file_name_split = file_name.split(config.config_decomposed["dataset_delimiter_file_name"])

            if len(file_name_split) < len(self.labels):
                logger.log_warn("Invalid data \"" + file_name + "\".")
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

        key = self.file_paths[index].split(".")[0].split(config.config_decomposed["dataset_delimiter_file_name"])[-1]
        label_original = self.classes_original.index(self.key_label_map[key])

        return (image, torch.LongTensor(labels), label_original)
