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

    def gatherDecomposedPredictionConfidences(self, outputs_decomposed):
        batch_size = outputs_decomposed[0].size(0)
        outputs_decomposed_gathered = torch.Tensor()
        outputs_decomposed_gathered = outputs_decomposed_gathered.to(self.device)

        for task_index in range(0, len(outputs_decomposed)):
            class_indices_task = self.class_indices_decomposed[:, task_index]
            class_indices_task = class_indices_task.repeat(1, batch_size)
            class_indices_task = class_indices_task.view(batch_size, len(self.dataset.classes_original), -1)

            outputs_decomposed[task_index] = self.softmax(outputs_decomposed[task_index])
            outputs_decomposed_task = outputs_decomposed[task_index].repeat(1, len(self.dataset.classes_original))
            outputs_decomposed_task = outputs_decomposed_task.view(batch_size, len(self.dataset.classes_original), -1)

            outputs_task = torch.gather(outputs_decomposed_task, 2, class_indices_task)
            outputs_decomposed_gathered = torch.cat([outputs_decomposed_gathered, outputs_task], 2)

        return outputs_decomposed_gathered

    def compose(self, outputs_decomposed):
        outputs_decomposed_gathered = self.gatherDecomposedPredictionConfidences(outputs_decomposed)
        output_composed = torch.prod(outputs_decomposed_gathered, 2)

        return output_composed
