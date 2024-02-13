import abc
import header
import json
import logger
import torch
import torchvision
import type

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        return

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def get_parameters(self):
        pass

class MLPSet(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.model_list = []

        for attribute in config_dataset["attributes"]:
            attribute_labels = attribute["labels"]
            attribute_labels.remove("")

            input_size = header.config_decomposed["model_input_height"] * header.config_decomposed["model_input_width"] * header.config_decomposed["model_input_channels"]
            hidden_size = header.config_decomposed["head_hidden_size"]
            output_size = len(attribute_labels)

            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size)
            )

            self.model_list.append(model)

        self.model_list = torch.nn.ModuleList(self.model_list)

        return

    def forward(self, input):
        outputs = []

        for model in self.model_list:
            outputs.append(model(input))

        return (outputs, None)

    def get_parameters(self):
        parameters = []

        for model in self.model_list:
            parameters.append(model.parameters())

        return parameters

class ResNet152(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_baseline["model_pretrained_weights"])

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        class_count = len(config_dataset["mappings"].keys())
        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(net_fc_in_features, class_count)

        return

    def forward(self, input):
        return self.net(input)

    def get_parameters(self):
        if header.config_baseline["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.fc.parameters()

class ResNet152MTL(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_decomposed["model_pretrained_weights"])

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Identity()

        layer_list = []

        for dataset_entry in config_dataset["attributes"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]
            head_hidden_size = header.config_decomposed["head_hidden_size"]

            dataset_labels.remove("")

            layers_hidden = torch.nn.Sequential(torch.nn.Linear(net_fc_in_features, head_hidden_size), torch.nn.ReLU())
            layer_final = torch.nn.Linear(head_hidden_size, len(dataset_labels))

            layer_dict = torch.nn.ModuleDict({"hidden": layers_hidden, "final": layer_final})
            layer_dict_task = torch.nn.ModuleDict({dataset_name: layer_dict})

            layer_list.append(layer_dict_task)

        self.net.heads_mtl = torch.nn.ModuleList(layer_list)

        return

    def forward(self, input):
        outputs_head = []
        outputs_head_hidden = []
        output_neck = self.net(input)

        for head in self.net.heads_mtl:
            head_layer_dict = list(head.values())[0]

            output_head_hidden = head_layer_dict["hidden"](output_neck)
            output_head = head_layer_dict["final"](output_head_hidden)

            outputs_head.append(output_head)
            outputs_head_hidden.append(output_head_hidden)

        return (outputs_head, outputs_head_hidden)

    def get_parameters(self):
        if header.config_decomposed["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads_mtl.parameters()

class ViTB32(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.vit_b_32(weights = header.config_baseline["model_pretrained_weights"])

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        for parameter in self.net.heads.parameters():
            parameter.requires_grad = True

        class_count = len(config_dataset["mappings"].keys())
        net_heads_head_in_features = self.net.heads.head.in_features
        self.net.heads.head = torch.nn.Linear(net_heads_head_in_features, class_count)

        return

    def forward(self, input):
        return self.net(input)

    def get_parameters(self):
        if header.config_baseline["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads.parameters()

class ViTB32MTL(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.vit_b_32(weights = header.config_decomposed["model_pretrained_weights"])

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        for parameter in self.net.heads.parameters():
            parameter.requires_grad = True

        net_heads_head_in_features = self.net.heads.head.in_features
        self.net.heads.head = torch.nn.Identity()

        layer_list = []

        for dataset_entry in config_dataset["attributes"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]
            head_hidden_size = header.config_decomposed["head_hidden_size"]

            dataset_labels.remove("")

            layers_hidden = torch.nn.Sequential(torch.nn.Linear(net_heads_head_in_features, head_hidden_size), torch.nn.ReLU())
            layer_final = torch.nn.Linear(head_hidden_size, len(dataset_labels))

            layer_dict = torch.nn.ModuleDict({"hidden": layers_hidden, "final": layer_final})
            layer_dict_task = torch.nn.ModuleDict({dataset_name: layer_dict})

            layer_list.append(layer_dict_task)

        self.net.heads_mtl = torch.nn.ModuleList(layer_list)

        return

    def forward(self, input):
        outputs_head = []
        outputs_head_hidden = []
        output_neck = self.net(input)

        for head in self.net.heads_mtl:
            head_layer_dict = list(head.values())[0]

            output_head_hidden = head_layer_dict["hidden"](output_neck)
            output_head = head_layer_dict["final"](output_head_hidden)

            outputs_head.append(output_head)
            outputs_head_hidden.append(output_head_hidden)

        return (outputs_head, outputs_head_hidden)

    def get_parameters(self):
        if header.config_decomposed["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads_mtl.parameters()

def createModelBaseline(device):
    file_config_dataset = open(header.file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    if header.config_baseline["model"] == type.ModelBaseline.resnet152.name:
        header.config_baseline["model_pretrained_weights"] = "IMAGENET1K_V2"
        logger.log_trace("Model pretrained weights: \"" + header.config_baseline["model_pretrained_weights"] + "\".")
        return ResNet152(config_dataset, device)
    elif header.config_baseline["model"] == type.ModelBaseline.vit_b_32.name:
        header.config_baseline["model_pretrained_weights"] = "IMAGENET1K_V1"
        logger.log_trace("Model pretrained weights: \"" + header.config_baseline["model_pretrained_weights"] + "\".")
        return ViTB32(config_dataset, device)
    else:
        logger.log_fatal("Unknown baseline network model \"" + header.config_baseline["model"] + "\".")
        exit(-1)

def createModelDecomposed(device):
    file_config_dataset = open(header.file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    if header.config_decomposed["model"] == type.ModelDecomposed.mlp_set.name:
        return MLPSet(config_dataset, device)
    elif header.config_decomposed["model"] == type.ModelDecomposed.resnet152_mtl.name:
        header.config_decomposed["model_pretrained_weights"] = "IMAGENET1K_V2"
        logger.log_trace("Model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
        return ResNet152MTL(config_dataset, device)
    elif header.config_decomposed["model"] == type.ModelDecomposed.vit_b_32_mtl.name:
        header.config_decomposed["model_pretrained_weights"] = "IMAGENET1K_V1"
        logger.log_trace("Model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
        return ViTB32MTL(config_dataset, device)
    else:
        logger.log_fatal("Unknown decomposed network model \"" + header.config_decomposed["model"] + "\".")
        exit(-1)
