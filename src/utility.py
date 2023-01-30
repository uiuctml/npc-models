
import header
import json
import logger
import math
import matplotlib.pyplot as plt
import os
import torch

def load(model, optimizer, learning_rate_scheduler):
    accuracy_validation = None
    criterion = None
    data_loader_train = None
    data_loader_validation = None
    epoch = None
    statistics = None

    if os.path.isdir(header.train_model_dir):
        if os.path.isfile(header.train_model_file_path_model):
            model.load_state_dict(torch.load(header.train_model_file_path_model))
            logger.log_info("Loaded model state from \"" + header.train_model_dir + "\".")

        if os.path.isfile(header.train_model_file_path_data_loader_train):
            data_loader_train = torch.load(header.train_model_file_path_data_loader_train)
            logger.log_info("Loaded training data loader from \"" + header.train_model_dir + "\".")

        if os.path.isfile(header.train_model_file_path_data_loader_validation):
            data_loader_validation = torch.load(header.train_model_file_path_data_loader_validation)
            logger.log_info("Loaded validation data loader from \"" + header.train_model_dir + "\".")

        if os.path.isfile(header.train_model_file_path_epoch):
            epoch = torch.load(header.train_model_file_path_epoch)
            logger.log_info("Loaded epoch from \"" + header.train_model_dir + "\".")

        if os.path.isfile(header.train_model_file_path_criterion):
            criterion = torch.load(header.train_model_file_path_criterion)
            logger.log_info("Loaded criterion from \"" + header.train_model_dir + "\".")

        if os.path.isfile(header.train_model_file_path_optimizer):
            optimizer.load_state_dict(torch.load(header.train_model_file_path_optimizer))
            logger.log_info("Loaded optimizer state from \"" + header.train_model_dir + "\".")

        if os.path.isfile(header.train_model_file_path_learning_rate_scheduler):
            learning_rate_scheduler.load_state_dict(torch.load(header.train_model_file_path_learning_rate_scheduler))
            logger.log_info("Loaded learning rate scheduler state from \"" + header.train_model_dir + "\".")

        if os.path.isfile(header.train_model_file_path_accuracy_validation):
            accuracy_validation = torch.load(header.train_model_file_path_accuracy_validation)
            logger.log_info("Loaded highest accuracy validation from \"" + header.train_model_dir + "\".")

        if os.path.isfile(header.train_model_file_path_statistics):
            with open(header.train_model_file_path_statistics, "r") as file_statistics:
                statistics = json.load(file_statistics)
                logger.log_info("Loaded statistics from \"" + header.train_model_dir + "\".")

    return (data_loader_train, data_loader_validation, epoch, criterion, accuracy_validation, statistics)

def plotStatistics(statistics):
    if statistics == None:
        return

    accuracies_training = []
    accuracies_training_xticks = [0]
    accuracies_validation = []
    accuracies_validation_xticks = [0]
    losses_training = []
    losses_training_xticks = [0]
    losses_validation = []
    losses_validation_xticks = [0]

    for epoch in statistics.keys():
        accuracies_training += statistics[epoch]["training_accuracies"]
        accuracies_training_xticks.append(len(accuracies_training) + 1)

        accuracies_validation += statistics[epoch]["validation_accuracies"]
        accuracies_validation_xticks.append(len(accuracies_validation) + 1)

        losses_training += statistics[epoch]["training_losses"]
        losses_training_xticks.append(len(losses_training) + 1)

        losses_validation += statistics[epoch]["validation_losses"]
        losses_validation_xticks.append(len(losses_validation) + 1)

    figure = plt.figure(figsize = (2, 2))
    figure_manager = plt.get_current_fig_manager()

    figure.add_subplot(2, 2, 1)
    plt.plot(accuracies_training)
    plt.xticks(accuracies_training_xticks, range(0, len(accuracies_training_xticks)))
    plt.title("Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    figure.add_subplot(2, 2, 2)
    plt.plot(accuracies_validation)
    plt.xticks(accuracies_validation_xticks, range(0, len(accuracies_validation_xticks)))
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    figure.add_subplot(2, 2, 3)
    plt.plot(losses_training)
    plt.xticks(losses_training_xticks, range(0, len(losses_training_xticks)))
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    figure.add_subplot(2, 2, 4)
    plt.plot(losses_validation)
    plt.xticks(losses_validation_xticks, range(0, len(losses_validation_xticks)))
    plt.title("Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    figure_manager.full_screen_toggle()
    plt.show()

    return

def save(model, data_loader_train, data_loader_validation, epoch, criterion, optimizer, learning_rate_scheduler, accuracy_validation, statistics):
    if not os.path.isdir(header.train_model_dir):
        os.makedirs(header.train_model_dir, exist_ok = True)

    torch.save(model.state_dict(), header.train_model_file_path_model)
    torch.save(data_loader_train, header.train_model_file_path_data_loader_train)
    torch.save(data_loader_validation, header.train_model_file_path_data_loader_validation)
    torch.save(epoch, header.train_model_file_path_epoch)
    torch.save(criterion, header.train_model_file_path_criterion)
    torch.save(optimizer.state_dict(), header.train_model_file_path_optimizer)
    torch.save(learning_rate_scheduler.state_dict(), header.train_model_file_path_learning_rate_scheduler)
    torch.save(accuracy_validation, header.train_model_file_path_accuracy_validation)

    with open(header.train_model_file_path_statistics, "w") as file_statistics:
        json.dump(statistics, file_statistics, indent = 4)

    logger.log_info("Saved training states and statistics to \"" + header.train_model_dir + "\".")

    return

def viewDataset(dataset, data_loader):
    if header.log_level < logger.LogLevel.trace:
        return

    figure_rows = header.dataset_view_row_count
    figure_cols = math.ceil(header.train_data_loader_batch_size / header.dataset_view_row_count)
    figure = plt.figure(figsize = (figure_cols, figure_rows))
    figure_manager = plt.get_current_fig_manager()
    input, labels = next(iter(data_loader))
    input = input.numpy().transpose((0, 2, 3, 1))
    class_list = list(dataset.classes[label] for label in labels)

    for i in range(0, header.train_data_loader_batch_size):
        figure.add_subplot(figure_cols, figure_rows, i + 1)
        plt.title(class_list[i])
        plt.axis("off")
        plt.imshow(input[i].squeeze())

    figure_manager.full_screen_toggle()
    plt.show()

    return