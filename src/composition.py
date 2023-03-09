import header
import json
import logger
import os
import torch
import torch.nn

class Composition():
    def __init__(self, dataset, device):
        self.config_generate = {}
        self.class_indices_decomposed = []
        self.dataset = dataset
        self.device = device
        self.softmax = torch.nn.Softmax(dim = 1)

        file_config_generate = open(os.path.join(header.config_decomposed["dir_dataset_test"], header.config_decomposed["file_name_config_dataset_generation"]), "r")
        self.config_generate = json.load(file_config_generate)
        file_config_generate.close()

        self.gatherDecomposedClassIndices()

        return

    def gatherDecomposedClassIndices(self):
        for class_name_original in self.dataset.classes_original:
            class_indices_decomposed = []

            for (dataset_index, dataset) in enumerate(self.dataset.config["datasets"]):
                dataset_name = dataset["name"]
                class_name_decomposed = self.config_generate[class_name_original]["labels"][dataset_name]

                if class_name_decomposed == "":
                    logger.log_warn("\"" + class_name_original + "\" contains an empty label for dataset \"" + dataset_name + "\".")
                    class_name_decomposed = dataset_name + header.config_decomposed["dataset_delimiter_label"] + header.config_decomposed["dataset_label_undefined_keyword"]

                class_indices_decomposed.append(self.dataset.class_to_idx[dataset_index][class_name_decomposed])

            self.class_indices_decomposed.append(class_indices_decomposed)

        self.class_indices_decomposed = torch.LongTensor(self.class_indices_decomposed)
        self.class_indices_decomposed = self.class_indices_decomposed.to(self.device)

        return

    def compose(self, outputs_decomposed):
        batch_size = outputs_decomposed[0].size(0)
        output_composed = []

        for i in range (0, len(outputs_decomposed)):
            outputs_decomposed[i] = self.softmax(outputs_decomposed[i])

        output_composed = torch.Tensor()
        output_composed = output_composed.to(self.device)

        for (task_index, task_outputs) in enumerate(outputs_decomposed):
            task_class_indices = self.class_indices_decomposed[:, task_index]

            task_outputs_batch = torch.Tensor()
            task_outputs_batch = task_outputs_batch.to(self.device)

            for batch_index in range(0, batch_size):
                task_outputs = torch.gather(outputs_decomposed[task_index][batch_index], 0, task_class_indices)
                task_outputs_batch = torch.cat([task_outputs_batch, task_outputs], 0)

            task_outputs_batch = torch.unsqueeze(task_outputs_batch, 0)
            output_composed = torch.cat([output_composed, task_outputs_batch], 0)

        output_composed = torch.transpose(output_composed, 1, 0)
        output_composed = torch.prod(output_composed, 1)
        output_composed = output_composed.view(batch_size, len(self.dataset.classes_original))

        return output_composed
