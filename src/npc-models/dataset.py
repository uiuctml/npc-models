import header
import json
import logger
import os
import PIL.Image
import torch
import utility

class NPCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform = None):
        self.config = {}
        self.file_paths = []
        self.indices_attribute = []
        self.indices_class = []
        self.label_to_index_attribute = []
        self.label_to_index_class = {}
        self.labels_attribute = []
        self.labels_class = []
        self.root = root
        self.transform = transform

        if not os.path.isdir(root):
            logger.log_error("Invalid dataset directory \"" + root + "\".")
            exit(-1)

        logger.log_info("Loading dataset \"" + root + "\"...")

        file_config_dataset = open(header.dataset_config_file_path, "r")
        self.config = json.load(file_config_dataset)
        file_config_dataset.close()

        self.labels_attribute = utility.getLabelsAttribute(self.config)
        self.labels_class = utility.getLabelsClass(self.config)
        self.label_to_index_attribute = utility.getIndicesFromLabelsAttribute(self.labels_attribute)
        self.label_to_index_class = utility.getIndicesFromLabelsClass(self.labels_class)

        for _ in self.config["attributes"]:
            self.indices_attribute.append([])

        if "instance_wise" in self.config and self.config["instance_wise"]:
            for class_name in os.listdir(root):
                for file_name in os.listdir(os.path.join(root, class_name)):
                    image_name = os.path.join(class_name, file_name)
                    index_class = self.label_to_index_class[class_name]

                    self.file_paths.append(os.path.abspath(os.path.join(root, image_name)))
                    self.indices_class.append(index_class)

                    for (index_attribute_name, attribute) in enumerate(self.config["attributes"]):
                        attribute_name = attribute["name"]
                        attribute_categories = self.config["mappings"][image_name]["labels"][attribute_name]
                        count_categories = len(self.labels_attribute[attribute_name])
                        index_attribute_categories = [0 for _ in range(count_categories)]

                        if isinstance(attribute_categories, list):
                            probability = 1 / len(attribute_categories)

                            for attribute_category in attribute_categories:
                                index_category = self.label_to_index_attribute[attribute_name][attribute_category]
                                index_attribute_categories[index_category] = probability
                        else:
                            index_category = self.label_to_index_attribute[attribute_name][attribute_categories]
                            index_attribute_categories[index_category] = 1

                        self.indices_attribute[index_attribute_name].append(index_attribute_categories)
        else:
            for class_name in self.labels_class:
                for file_name in os.listdir(os.path.join(root, class_name)):
                    index_class = self.label_to_index_class[class_name]

                    self.file_paths.append(os.path.abspath(os.path.join(root, class_name, file_name)))
                    self.indices_class.append(index_class)

                    for (index_attribute_name, attribute) in enumerate(self.config["attributes"]):
                        attribute_name = attribute["name"]
                        attribute_categories = self.config["mappings"][class_name]["labels"][attribute_name]
                        count_categories = len(self.labels_attribute[attribute_name])
                        index_attribute_categories = [0 for _ in range(count_categories)]

                        if isinstance(attribute_categories, list):
                            probability = 1 / len(attribute_categories)

                            for attribute_category in attribute_categories:
                                index_category = self.label_to_index_attribute[attribute_name][attribute_category]
                                index_attribute_categories[index_category] = probability
                        else:
                            index_category = self.label_to_index_attribute[attribute_name][attribute_categories]
                            index_attribute_categories[index_category] = 1

                        self.indices_attribute[index_attribute_name].append(index_attribute_categories)

        return

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, instance):
        image = PIL.Image.open(self.file_paths[instance]).convert("RGB")
        indices_attribute = []

        if self.transform is not None:
            image = self.transform(image)

        for index_attribute in self.indices_attribute:
            indices_attribute.append(torch.Tensor(index_attribute[instance]))

        return (image, indices_attribute, self.indices_class[instance], self.file_paths[instance])
