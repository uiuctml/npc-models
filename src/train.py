#!/usr/bin/env python3

import header
import logger
import math
import matplotlib.pyplot as plt
import torch
import torch.nn
import torch.optim
import torchsummary
import torchvision
import tqdm

def train(model, data_loader, criterion, optimizer, learning_rate_scheduler, device):
    running_loss = 0
    running_corrects = 0

    model.train()

    for (input, labels) in data_loader:
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * input.size(0)
        running_corrects += torch.sum(predictions == labels.data)

    learning_rate_scheduler.step()

    return (running_loss, running_corrects)

def validate(model, data_loader, criterion, device):
    running_loss = 0
    running_corrects = 0

    model.eval()

    for (input, labels) in data_loader:
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            output = model(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)

        running_loss += loss.item() * input.size(0)
        running_corrects += torch.sum(predictions == labels.data)

    return (running_loss, running_corrects)

def save(model, epoch, criterion, optimizer):
    torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "criterion": criterion,
        }, header.model_file_name)
    
    return

def viewDataset(dataset, data_loader):
    if header.log_level < logger.LogLevel.trace:
        return

    figure_rows = header.dataset_view_row_count
    figure_cols = math.ceil(header.data_loader_batch_size / header.dataset_view_row_count)
    figure = plt.figure(figsize = (figure_cols, figure_rows))
    figure_manager = plt.get_current_fig_manager()
    input, labels = next(iter(data_loader))
    input = input.numpy().transpose((0, 2, 3, 1))
    class_list = list(dataset.classes[label] for label in labels)

    for i in range(0, header.data_loader_batch_size):
        figure.add_subplot(figure_cols, figure_rows, i + 1)
        plt.title(class_list[i])
        plt.axis("off")
        plt.imshow(input[i].squeeze())

    figure_manager.full_screen_toggle()
    plt.show()

    return

def main():
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.model_input_height, header.model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(root = header.dataset_dir_images, transform = dataset_transforms)
    dataset_split_lengths = [header.dataset_split_percentage_train, header.dataset_split_percentage_validation]
    (dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset, dataset_split_lengths)

    data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = header.data_loader_batch_size, shuffle = header.data_loader_shuffle, num_workers = header.data_loader_worker_count, pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = header.data_loader_batch_size, shuffle = header.data_loader_shuffle, num_workers = header.data_loader_worker_count, pin_memory = True)

    device = torch.device("cuda")

    model = torchvision.models.resnet18(weights = header.model_pretrained_weights)
    model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = header.optimizer_learning_rate, momentum = header.optimizer_momentum)
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = header.learning_rate_scheduler_step_size, gamma = header.learning_rate_scheduler_gamma)

    model_accuracy_validation = 0

    if header.log_level >= logger.LogLevel.debug:
        model_input_size = (header.model_input_channels, header.model_input_height, header.model_input_width)
        torchsummary.summary(model, input_size=model_input_size)

    viewDataset(dataset, data_loader_train)
    viewDataset(dataset, data_loader_validation)

    progress_bar_epoch = tqdm.tqdm(total = header.model_epochs, position = 0)
    progress_bar_loss_train = tqdm.tqdm(total = 1.0, position = 1)
    progress_bar_loss_validation = tqdm.tqdm(total = 1.0, position = 2)
    progress_bar_accuracy_train = tqdm.tqdm(total = 1.0, position = 3)
    progress_bar_accuracy_validation = tqdm.tqdm(total = 1.0, position = 4)

    progress_bar_epoch.set_description_str("Epoch")
    progress_bar_loss_train.set_description_str("Training loss")
    progress_bar_loss_validation.set_description_str("Validation loss")
    progress_bar_accuracy_train.set_description_str("Training accuracy")
    progress_bar_accuracy_validation.set_description_str("Validation accuracy")

    for epoch in range(header.model_epochs):
        stats_train = train(model, data_loader_train, criterion, optimizer, learning_rate_scheduler, device)
        stats_validation = validate(model, data_loader_validation, criterion, device)

        epoch_loss_train = stats_train[0] / len(dataset_subset_train)
        epoch_accuracy_train = stats_train[1] / len(dataset_subset_train)

        epoch_loss_validation = stats_validation[0] / len(dataset_subset_validation)
        epoch_accuracy_validation = stats_validation[1] / len(dataset_subset_validation)

        progress_bar_epoch.n = epoch
        progress_bar_loss_train.n = epoch_loss_train
        progress_bar_loss_validation.n = epoch_loss_validation
        progress_bar_accuracy_train.n = epoch_accuracy_train
        progress_bar_accuracy_validation.n = epoch_accuracy_validation

        progress_bar_epoch.refresh()
        progress_bar_loss_train.refresh()
        progress_bar_loss_validation.refresh()
        progress_bar_accuracy_train.refresh()
        progress_bar_accuracy_validation.refresh()

        if epoch_accuracy_validation > model_accuracy_validation:
            model_accuracy_validation = epoch_accuracy_validation
            save(model, epoch, criterion, optimizer)

    progress_bar_epoch.close()
    progress_bar_loss_train.close()
    progress_bar_loss_validation.close()
    progress_bar_accuracy_train.close()
    progress_bar_accuracy_validation.close()

    logger.log_info("Best validation accuracy: " + str(model_accuracy_validation) + ".")

    return

if __name__ == "__main__":
    main()
