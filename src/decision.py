import header
import json
import logger
import os
import torch
import torch.nn

class Decision():
    def __init__(self, dataset, device):
        self.config_generate = {}
        self.dataset = dataset
        self.device = device
        self.softmax = torch.nn.Softmax(dim = 1)

        file_config_generate = open(os.path.join(header.config_decomposed["dir_dataset"], header.config_decomposed["file_name_config_dataset_generation"]), "r")
        self.config_generate = json.load(file_config_generate)
        file_config_generate.close()

        return

    def make(self, outputs_task, batch_size):
        output = []

        for i in range (0, len(outputs_task)):
            outputs_task[i] = self.softmax(outputs_task[i])

        # Iterate through batches
        for batch_index in range(0, batch_size):
            output_batch = []

            # Iterate through original classes
            for class_name in self.config_generate.keys():
                prediction_class = 1

                # Iterate through tasks
                for (dataset_index, dataset_name) in enumerate(self.config_generate[class_name]["labels"].keys()):
                    label = self.config_generate[class_name]["labels"][dataset_name]

                    if label == "":
                        label = dataset_name + header.config_decomposed["dataset_delimiter_label"] + header.config_decomposed["dataset_label_undefined_keyword"]

                    label_index = self.dataset.classes[dataset_index].index(label)
                    prediction_class *= outputs_task[dataset_index][batch_index][label_index].item()

                output_batch.append(prediction_class)

            output.append(output_batch)

        output = torch.Tensor(output)
        output = output.to(self.device)

        return output
