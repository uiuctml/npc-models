import header
import logger
import torch.nn
import torchvision
import type

class ResNet152(torch.nn.Module):
    def __init__(self, class_count):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_baseline["model_pretrained_weights"])
        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(net_fc_in_features, class_count)

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.parameters():
                parameter.requires_grad = False

        return

    def forward(self, input):
        return self.net(input)

class ResNet152Dropout(torch.nn.Module):
    def __init__(self, class_count):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_baseline["model_pretrained_weights"])
        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(torch.nn.Dropout(p = header.config_baseline["train_dropout_probability"]), torch.nn.Linear(net_fc_in_features, class_count))

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.parameters():
                parameter.requires_grad = False

        return

    def forward(self, input):
        return self.net(input)

class ViTB32(torch.nn.Module):
    def __init__(self, class_count):
        super().__init__()

        self.net = torchvision.models.vit_b_32(weights = header.config_baseline["model_pretrained_weights"])
        net_heads_head_in_features = self.net.heads.head.in_features
        self.net.heads.head = torch.nn.Linear(net_heads_head_in_features, class_count)

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.parameters():
                parameter.requires_grad = False

        return

    def forward(self, input):
        return self.net(input)

class ResNet152MTL(torch.nn.Module):
    def __init__(self, config_dataset):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_decomposed["model_pretrained_weights"])
        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Identity()

        layer_list = []

        for dataset_entry in config_dataset["datasets"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]

            layer = torch.nn.Linear(net_fc_in_features, len(dataset_labels))
            layer_dict = torch.nn.ModuleDict({dataset_name: layer})

            layer_list.append(layer_dict)

        self.net.heads_mtl = torch.nn.ModuleList(layer_list)

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.parameters():
                parameter.requires_grad = False

        return

    def forward(self, input):
        outputs_head = []
        output_neck = self.net(input)

        for head in self.net.heads_mtl:
            head_layer = list(head.values())[0]
            outputs_head.append(head_layer(output_neck))

        return outputs_head

class ViTB32MTL(torch.nn.Module):
    def __init__(self, config_dataset):
        super().__init__()

        self.net = torchvision.models.vit_b_32(weights = header.config_baseline["model_pretrained_weights"])
        net_heads_head_in_features = self.net.heads.head.in_features
        self.net.heads.head = torch.nn.Identity()

        layer_list = []

        for dataset_entry in config_dataset["datasets"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]

            layer = torch.nn.Linear(net_heads_head_in_features, len(dataset_labels))
            layer_dict = torch.nn.ModuleDict({dataset_name: layer})

            layer_list.append(layer_dict)

        self.net.heads_mtl = torch.nn.ModuleList(layer_list)

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.parameters():
                parameter.requires_grad = False

        return

    def forward(self, input):
        outputs_head = []
        output_neck = self.net(input)

        for head in self.net.heads_mtl:
            head_layer = list(head.values())[0]
            outputs_head.append(head_layer(output_neck))

        return outputs_head

def createModelBaseline(*args, **kwargs):
    if header.config_baseline["model"] == type.NetworkModelBaseline.resnet152.name:
        return ResNet152(*args, **kwargs)
    elif header.config_baseline["model"] == type.NetworkModelBaseline.resnet152_dropout.name:
        return ResNet152Dropout(*args, **kwargs)
    elif header.config_baseline["model"] == type.NetworkModelBaseline.vit_b_32.name:
        return ViTB32(*args, **kwargs)
    else:
        logger.log_warn("Unknown baseline network model \"" + header.config_baseline["model"] + "\". Using \"" + type.NetworkModelBaseline.resnet152.name + "\".")
        return ResNet152(*args, **kwargs)

def createModelDecomposed(*args, **kwargs):
    if header.config_decomposed["model"] == type.NetworkModelDecomposed.resnet152_mtl.name:
        return ResNet152MTL(*args, **kwargs)
    elif header.config_decomposed["model"] == type.NetworkModelDecomposed.vit_b_32_mtl.name:
        return ViTB32MTL(*args, **kwargs)
    else:
        logger.log_warn("Unknown decomposed network model \"" + header.config_baseline["model"] + "\". Using \"" + type.NetworkModelDecomposed.resnet152_mtl.name + "\".")
        return ResNet152MTL(*args, **kwargs)
