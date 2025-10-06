import abc
import header
import logger
import torch
import torch_explain
import torchvision
import type
import utility

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

class ABM(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        labels_attribute = utility.getLabelsAttribute(config_dataset)
        labels_class = utility.getLabelsClass(config_dataset)
        labels_categories = []

        for attribute_name in labels_attribute.keys():
            labels_categories += labels_attribute[attribute_name]

        self.net = ResNet34MTL(config_dataset, device)
        self.net.head = torch.nn.Linear(len(labels_categories), len(labels_class))

        return

    def forward(self, input):
        (output_neck, _) = self.net(input)
        output_head = self.net.head(torch.cat(output_neck, dim = 1))

        return (output_neck, output_head)

    def get_parameters(self):
        return self.net.parameters()

class CBM(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        labels_attribute = utility.getLabelsAttribute(config_dataset)
        labels_class = utility.getLabelsClass(config_dataset)
        labels_categories = []

        for attribute_name in labels_attribute.keys():
            labels_categories += labels_attribute[attribute_name]

        self.net = torchvision.models.resnet34(weights = "IMAGENET1K_V1")
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, len(labels_categories))
        self.net.head = torch.nn.Linear(len(labels_categories), len(labels_class))

        return

    def forward(self, input):
        output_neck = self.net(input)
        output_head = self.net.head(output_neck)

        return (output_neck, output_head)

    def get_parameters(self):
        return self.net.parameters()

class CEM(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        labels_attribute = utility.getLabelsAttribute(config_dataset)
        labels_class = utility.getLabelsClass(config_dataset)
        labels_categories = []

        for attribute_name in labels_attribute.keys():
            labels_categories += labels_attribute[attribute_name]

        self.net = torchvision.models.resnet34(weights = "IMAGENET1K_V1")
        self.net.fc = torch_explain.nn.ConceptEmbedding(self.net.fc.in_features, len(labels_categories), header.config_baseline["model_embedding_size"])
        self.net.head = torch.nn.Linear(len(labels_categories) * header.config_baseline["model_embedding_size"], len(labels_class))

        return

    def forward(self, input):
        (concept_embedding, output_neck) = self.net(input)
        output_head = self.net.head(concept_embedding.reshape(len(concept_embedding), -1))

        return (output_neck, output_head)

    def get_parameters(self):
        return self.net.parameters()

class DCR(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        labels_attribute = utility.getLabelsAttribute(config_dataset)
        labels_class = utility.getLabelsClass(config_dataset)
        labels_categories = []

        for attribute_name in labels_attribute.keys():
            labels_categories += labels_attribute[attribute_name]

        self.net = torchvision.models.resnet34(weights = "IMAGENET1K_V1")
        self.net.fc = torch_explain.nn.ConceptEmbedding(self.net.fc.in_features, len(labels_categories), header.config_baseline["model_embedding_size"])
        self.net.head = torch_explain.nn.concepts.ConceptReasoningLayer(header.config_baseline["model_embedding_size"], len(labels_class))

        return

    def forward(self, input):
        (concept_embedding, output_neck) = self.net(input)
        output_head = self.net.head(concept_embedding, output_neck)

        return (output_neck, output_head)

    def get_parameters(self):
        return self.net.parameters()

class ResNet34(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.resnet34(weights = "IMAGENET1K_V1")
        class_count = len(utility.getLabelsClass(config_dataset))
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, class_count)

        return

    def forward(self, input):
        return self.net(input)

    def get_parameters(self):
        return self.net.parameters()

class ResNet34MTL(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        layer_list = []
        self.net = torchvision.models.resnet34(weights = "IMAGENET1K_V1")

        for attribute in config_dataset["attributes"]:
            attribute_name = attribute["name"]
            attribute_labels = attribute["labels"]

            if "" in attribute_labels:
                attribute_labels.remove("")

            layers_hidden = torch.nn.Sequential(torch.nn.Linear(self.net.fc.in_features, header.config_neural["model_head_hidden_size"]), torch.nn.ReLU())
            layer_final = torch.nn.Linear(header.config_neural["model_head_hidden_size"], len(attribute_labels))

            layer_dict = torch.nn.ModuleDict({"hidden": layers_hidden, "final": layer_final})
            layer_dict_task = torch.nn.ModuleDict({attribute_name: layer_dict})

            layer_list.append(layer_dict_task)

        self.net.fc = torch.nn.Identity()
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
        return self.net.parameters()

def createModelReference(config_dataset, device):
    if header.config_baseline["model"] == type.ModelReference.abm.name:
        return ABM(config_dataset, device)
    elif header.config_baseline["model"] == type.ModelReference.cbm.name:
        return CBM(config_dataset, device)
    elif header.config_baseline["model"] == type.ModelReference.cem.name:
        return CEM(config_dataset, device)
    elif header.config_baseline["model"] == type.ModelReference.dcr.name:
        return DCR(config_dataset, device)
    else:
        logger.log_fatal("Unknown reference model \"" + header.config_baseline["model"] + "\".")
        exit(-1)
