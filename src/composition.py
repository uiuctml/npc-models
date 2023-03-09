import header
import json
import os
import torch
import torch.nn

class Composition():
    def __init__(self, dataset, device):
        self.config_generate = {}
        self.dataset = dataset
        self.device = device
        self.softmax = torch.nn.Softmax(dim = 1)

        file_config_generate = open(os.path.join(header.config_decomposed["dir_dataset_test"], header.config_decomposed["file_name_config_dataset_generation"]), "r")
        self.config_generate = json.load(file_config_generate)
        file_config_generate.close()

        return

    def compose(self, outputs_decomposed):
        batch_size = outputs_decomposed[0].size(0)
        output_composed = []

        # Apply softmax to decomposed outputs
        for i in range (0, len(outputs_decomposed)):
            outputs_decomposed[i] = self.softmax(outputs_decomposed[i])

        class_indices_decomposed_map = []

        # construct the index map
        for class_name_original in self.dataset.classes_original:
            prediction_original = 1

            class_indices_decomposed = []
            for (dataset_index, dataset) in enumerate(self.dataset.config["datasets"]):
                dataset_name = dataset["name"]
                class_name_decomposed = self.config_generate[class_name_original]["labels"][dataset_name]

                if class_name_decomposed == "":
                    class_name_decomposed = dataset_name + header.config_decomposed["dataset_delimiter_label"] + header.config_decomposed["dataset_label_undefined_keyword"]

                class_indices_decomposed.append(self.dataset.class_to_idx[dataset_index][class_name_decomposed])

            class_indices_decomposed_map.append(class_indices_decomposed)

        class_indices_decomposed_map = torch.LongTensor(class_indices_decomposed_map)
        class_indices_decomposed_map = class_indices_decomposed_map.to(self.device)

        output_composed = torch.Tensor()
        output_composed = output_composed.to(self.device)

        # construct the output tensor 1 (401*batch_size * 4)
        for (task_index, task_outputs) in enumerate(outputs_decomposed):
            task_class_indices = class_indices_decomposed_map[:, task_index]

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

        # # Iterate through batches
        # for batch_index in range(0, batch_size):
        #     output_composed_batch = []
        #
        #     # Iterate through original classes
        #     for class_name_original in self.dataset.classes_original:
        #         prediction_original = 1
        #
        #         # Iterate through decomposed dataset
        #         for (dataset_index, dataset) in enumerate(self.dataset.config["datasets"]):
        #             dataset_name = dataset["name"]
        #             class_name_decomposed = self.config_generate[class_name_original]["labels"][dataset_name]
        #
        #             if class_name_decomposed == "":
        #                 class_name_decomposed = dataset_name + header.config_decomposed["dataset_delimiter_label"] + header.config_decomposed["dataset_label_undefined_keyword"]
        #
        #             class_index_decomposed = self.dataset.class_to_idx[dataset_index][class_name_decomposed]
        #             prediction_original *= outputs_decomposed[dataset_index][batch_index][class_index_decomposed].item()
        #
        #         output_composed_batch.append(prediction_original)
        #
        #     output_composed.append(output_composed_batch)
        # 
        # output_composed = torch.Tensor(output_composed)
        # output_composed = output_composed.to(self.device)

        return output_composed
