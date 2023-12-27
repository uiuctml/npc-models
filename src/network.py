import header
import logger
import torch.nn
import torchvision
import type
import clip

class ResNet101CLIP(torch.nn.Module):
    def __init__(self, class_count, device):
        super().__init__()

        self.net = clip.load(name = "RN101", device = device)[0]

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        net_fc_in_features = self.net.visual.output_dim
        self.net.fc = torch.nn.Linear(net_fc_in_features, class_count)

        return

    def forward(self, input):
        output_neck = self.net.encode_image(input).type(self.net.fc.weight.dtype)

        return self.net.fc(output_neck)

    def getOptimizerParameters(self):
        if header.config_baseline["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.fc.parameters()

class ResNet152(torch.nn.Module):
    def __init__(self, class_count, device):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_baseline["model_pretrained_weights"])

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(net_fc_in_features, class_count)

        return

    def forward(self, input):
        return self.net(input)

    def getOptimizerParameters(self):
        if header.config_baseline["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.fc.parameters()

class ResNet152Dropout(torch.nn.Module):
    def __init__(self, class_count, device):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_baseline["model_pretrained_weights"])

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(torch.nn.Dropout(p = header.config_baseline["train_dropout_probability"]), torch.nn.Linear(net_fc_in_features, class_count))

        return

    def forward(self, input):
        return self.net(input)

    def getOptimizerParameters(self):
        if header.config_baseline["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.fc.parameters()

class ViTB32(torch.nn.Module):
    def __init__(self, class_count, device):
        super().__init__()

        self.net = torchvision.models.vit_b_32(weights = header.config_baseline["model_pretrained_weights"])

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        for parameter in self.net.heads.parameters():
            parameter.requires_grad = True

        net_heads_head_in_features = self.net.heads.head.in_features
        self.net.heads.head = torch.nn.Linear(net_heads_head_in_features, class_count)

        return

    def forward(self, input):
        return self.net(input)

    def getOptimizerParameters(self):
        if header.config_baseline["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads.parameters()

class ViTB32CLIP(torch.nn.Module):
    def __init__(self, class_count, device):
        super().__init__()

        self.net = clip.load(name = "ViT-B/32", device = device)[0]

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        net_fc_in_features = self.net.visual.output_dim
        self.net.fc = torch.nn.Linear(net_fc_in_features, class_count)

        return

    def forward(self, input):
        output_neck = self.net.encode_image(input).type(self.net.fc.weight.dtype)
        return self.net.fc(output_neck)

    def getOptimizerParameters(self):
        if header.config_baseline["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.fc.parameters()

class ResNet101CLIPMTL(torch.nn.Module):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = clip.load(name = "RN101", device = device)[0]

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        net_fc_in_features = self.net.visual.output_dim

        layer_list = []

        for dataset_entry in config_dataset["datasets"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]

            layer = torch.nn.Linear(net_fc_in_features, len(dataset_labels))
            layer_dict = torch.nn.ModuleDict({dataset_name: layer})
            self.dtype = layer.weight.dtype

            layer_list.append(layer_dict)

        self.net.heads_mtl = torch.nn.ModuleList(layer_list)

        return

    def forward(self, input):
        outputs_head = []
        output_neck = self.net.encode_image(input).type(self.dtype)

        for head in self.net.heads_mtl:
            head_layer = list(head.values())[0]
            outputs_head.append(head_layer(output_neck))

        return outputs_head

    def getOptimizerParameters(self):
        if header.config_decomposed["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads_mtl.parameters()

class ResNet152MTL(torch.nn.Module):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_decomposed["model_pretrained_weights"])

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Identity()

        layer_list = []

        for dataset_entry in config_dataset["datasets"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]
            head_hidden_size = header.config_decomposed["head_hidden_sizes"][dataset_name]

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

    def getOptimizerParameters(self):
        if header.config_decomposed["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads_mtl.parameters()

class ViTB32MTL(torch.nn.Module):
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

        for dataset_entry in config_dataset["datasets"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]
            head_hidden_size = header.config_decomposed["head_hidden_sizes"][dataset_name]

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

    def getOptimizerParameters(self):
        if header.config_decomposed["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads_mtl.parameters()

class ViTB32CLIPMTL(torch.nn.Module):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = self.net = clip.load(name = "ViT-B/32", device = device)[0]

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        net_fc_in_features = self.net.visual.output_dim

        layer_list = []

        for dataset_entry in config_dataset["datasets"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]
            head_hidden_size = header.config_decomposed["head_hidden_sizes"][dataset_name]

            layers_hidden = torch.nn.Sequential(torch.nn.Linear(net_fc_in_features, head_hidden_size), torch.nn.ReLU())
            layer_final = torch.nn.Linear(head_hidden_size, len(dataset_labels))
            self.dtype = layer_final.weight.dtype

            layer_dict = torch.nn.ModuleDict({"hidden": layers_hidden, "final": layer_final})
            layer_dict_task = torch.nn.ModuleDict({dataset_name: layer_dict})

            layer_list.append(layer_dict_task)

        self.net.heads_mtl = torch.nn.ModuleList(layer_list)

        return

    def forward(self, input):
        outputs_head = []
        outputs_head_hidden = []
        output_neck = self.net.encode_image(input).type(self.dtype)

        for head in self.net.heads_mtl:
            head_layer_dict = list(head.values())[0]

            output_head_hidden = head_layer_dict["hidden"](output_neck)
            output_head = head_layer_dict["final"](output_head_hidden)

            outputs_head.append(output_head)
            outputs_head_hidden.append(output_head_hidden)

        return (outputs_head, outputs_head_hidden)

    def getOptimizerParameters(self):
        if header.config_decomposed["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads_mtl.parameters()

def createModelBaseline(class_count, device):
    if header.config_baseline["model"] == type.NetworkModelBaseline.resnet101_clip.name:
        logger.log_trace("Model pretrained weights: \"" + header.config_baseline["model_pretrained_weights"] + "\".")
        return ResNet101CLIP(class_count, device)
    elif header.config_baseline["model"] == type.NetworkModelBaseline.resnet152.name:
        header.config_baseline["model_pretrained_weights"] = "IMAGENET1K_V2"
        logger.log_trace("Model pretrained weights: \"" + header.config_baseline["model_pretrained_weights"] + "\".")
        return ResNet152(class_count, device)
    elif header.config_baseline["model"] == type.NetworkModelBaseline.resnet152_dropout.name:
        header.config_baseline["model_pretrained_weights"] = "IMAGENET1K_V2"
        logger.log_trace("Model pretrained weights: \"" + header.config_baseline["model_pretrained_weights"] + "\".")
        return ResNet152Dropout(class_count, device)
    elif header.config_baseline["model"] == type.NetworkModelBaseline.vit_b_32.name:
        header.config_baseline["model_pretrained_weights"] = "IMAGENET1K_V1"
        logger.log_trace("Model pretrained weights: \"" + header.config_baseline["model_pretrained_weights"] + "\".")
        return ViTB32(class_count, device)
    elif header.config_baseline["model"] == type.NetworkModelBaseline.vit_b_32_clip.name:
        logger.log_trace("Model pretrained weights: \"" + header.config_baseline["model_pretrained_weights"] + "\".")
        return ViTB32CLIP(class_count, device)
    else:
        logger.log_fatal("Unknown baseline network model \"" + header.config_baseline["model"] + "\".")
        exit(1)

def createModelDecomposed(config_dataset, device):
    if header.config_decomposed["model"] == type.NetworkModelDecomposed.resnet101_clip_mtl.name:
        logger.log_trace("Model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
        return ResNet101CLIPMTL(config_dataset, device)
    elif header.config_decomposed["model"] == type.NetworkModelDecomposed.resnet152_mtl.name:
        header.config_decomposed["model_pretrained_weights"] = "IMAGENET1K_V2"
        logger.log_trace("Model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
        return ResNet152MTL(config_dataset, device)
    elif header.config_decomposed["model"] == type.NetworkModelDecomposed.vit_b_32_mtl.name:
        header.config_decomposed["model_pretrained_weights"] = "IMAGENET1K_V1"
        logger.log_trace("Model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
        return ViTB32MTL(config_dataset, device)
    elif header.config_decomposed["model"] == type.NetworkModelDecomposed.vit_b_32_clip_mtl.name:
        logger.log_trace("Model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
        return ViTB32CLIPMTL(config_dataset, device)
    else:
        logger.log_fatal("Unknown decomposed network model \"" + header.config_decomposed["model"] + "\".")
        exit(1)
