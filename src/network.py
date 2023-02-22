import torch.nn
import torchvision
import wandb

class BaselineNetworkA(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.net = torchvision.models.resnet152(weights = wandb.config.model_pretrained_weights)
        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(net_fc_in_features, len(dataset.classes))

        return

    def forward(self, input):
        return self.net(input)

class BaselineNetworkB(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.net = torchvision.models.resnet152(weights = wandb.config.model_pretrained_weights)
        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(torch.nn.Dropout(p = wandb.config.train_dropout_probability), torch.nn.Linear(net_fc_in_features, len(dataset.classes)))

        return

    def forward(self, input):
        return self.net(input)

class DecomposedNetworkA(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.net = torchvision.models.resnet152(weights = wandb.config.model_pretrained_weights)
        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Identity()

        layer_list = []

        for dataset_entry in dataset.config["datasets"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]

            layer = torch.nn.Linear(net_fc_in_features, len(dataset_labels))
            layer_dict = torch.nn.ModuleDict({dataset_name: layer})

            layer_list.append(layer_dict)

        self.net.heads = torch.nn.ModuleList(layer_list)

        return

    def forward(self, input):
        outputs_head = []
        output_neck = self.net(input)

        for head in self.net.heads:
            head_layer = list(head.values())[0]
            outputs_head.append(head_layer(output_neck))

        return outputs_head
