import json
import logger
import os
import torch
import torch.nn
import wandb

class Decision():
    def __init__(self, dataset, device):
        self.classes = []
        self.config = {}
        self.dataset = dataset
        self.device = device
        self.label_map = {}
        self.softmax = torch.nn.Softmax(dim = 1)

        file_config_dataset_generation = open(os.path.join(wandb.config.dir_dataset, wandb.config.file_name_config_dataset_generation), "r")
        self.config = json.load(file_config_dataset_generation)
        file_config_dataset_generation.close()

        for class_name in self.config.keys():
            self.classes.append(class_name)

        self.constructLabelMap()

        return

    def constructLabelMap(self):
        self.label_map = {}

        for class_name in self.config.keys():
            labels = ""

            for (dataset_index, dataset_name) in enumerate(self.config[class_name]["labels"].keys()):
                label = self.config[class_name]["labels"][dataset_name]

                if label == "":
                    logger.log_warn("\"" + class_name + "\" contains an empty label for dataset \"" + dataset_name + "\".")
                    label = dataset_name + wandb.config.dataset_delimiter_label + wandb.config.dataset_label_undefined_keyword

                if labels != "":
                    labels += wandb.config.dataset_delimiter_file_name
                
                labels += str(self.dataset.classes[dataset_index].index(label))
            
            self.label_map[labels] = self.classes.index(class_name)

        return

    def make(self, outputs_task, labels_task, batch_size):
        torch.set_printoptions(sci_mode = False)

        labels = self.mapLabels(labels_task)
        output = []

        for i in range (0, len(outputs_task)):
            outputs_task[i] = self.softmax(outputs_task[i])

        # Iterate through batches
        for batch_index in range(0, batch_size):
            output_batch = []

            # Iterate through original classes
            for class_name in self.config.keys():
                prediction_class = 1

                # Iterate through tasks
                for (dataset_index, dataset_name) in enumerate(self.config[class_name]["labels"].keys()):
                    label = self.config[class_name]["labels"][dataset_name]

                    if label == "":
                        label = dataset_name + wandb.config.dataset_delimiter_label + wandb.config.dataset_label_undefined_keyword

                    label_index = self.dataset.classes[dataset_index].index(label)
                    prediction_class *= outputs_task[dataset_index][batch_index][label_index].item()

                output_batch.append(prediction_class)

            output.append(output_batch)

        output = torch.Tensor(output)
        output = output.to(self.device)

        return (output, labels)

    def mapLabels(self, labels):
        labels_mapped = []

        for labels_batch in labels:
            labels_key = ""

            for label in labels_batch:
                if labels_key != "":
                    labels_key += wandb.config.dataset_delimiter_file_name

                labels_key += str(label.item())
            
            labels_mapped.append(self.label_map[labels_key])

        labels_mapped = torch.LongTensor(labels_mapped)
        labels_mapped = labels_mapped.to(self.device)

        return labels_mapped
