
import header
import logger
import math
import matplotlib.pyplot as plt
import os
import torch

def load(model, optimizer, learning_rate_scheduler):
    data_loader_train = None
    data_loader_validation = None
    epoch = None
    criterion = None
    accuracy_validation = None

    if os.path.isdir(header.model_dir):
        if os.path.isfile(header.model_file_path_model):
            model.load_state_dict(torch.load(header.model_file_path_model))
            logger.log_info("Loaded model state from \"" + header.model_dir + "\".")

        if os.path.isfile(header.model_file_path_data_loader_train):
            data_loader_train = torch.load(header.model_file_path_data_loader_train)
            logger.log_info("Loaded training data loader from \"" + header.model_dir + "\".")

        if os.path.isfile(header.model_file_path_data_loader_validation):
            data_loader_validation = torch.load(header.model_file_path_data_loader_validation)
            logger.log_info("Loaded validation data loader from \"" + header.model_dir + "\".")

        if os.path.isfile(header.model_file_path_epoch):
            epoch = torch.load(header.model_file_path_epoch)
            logger.log_info("Loaded epoch from \"" + header.model_dir + "\".")

        if os.path.isfile(header.model_file_path_criterion):
            criterion = torch.load(header.model_file_path_criterion)
            logger.log_info("Loaded criterion from \"" + header.model_dir + "\".")

        if os.path.isfile(header.model_file_path_optimizer):
            optimizer.load_state_dict(torch.load(header.model_file_path_optimizer))
            logger.log_info("Loaded optimizer state from \"" + header.model_dir + "\".")

        if os.path.isfile(header.model_file_path_learning_rate_scheduler):
            learning_rate_scheduler.load_state_dict(torch.load(header.model_file_path_learning_rate_scheduler))
            logger.log_info("Loaded learning rate scheduler state from \"" + header.model_dir + "\".")

        if os.path.isfile(header.model_file_path_accuracy_validation):
            accuracy_validation = torch.load(header.model_file_path_accuracy_validation)
            logger.log_info("Loaded highest accuracy validation from \"" + header.model_dir + "\".")

    return (data_loader_train, data_loader_validation, epoch, criterion, accuracy_validation)

def save(model, data_loader_train, data_loader_validation, epoch, criterion, optimizer, learning_rate_scheduler, accuracy_validation):
    if not os.path.isdir(header.model_dir):
        os.makedirs(header.model_dir, exist_ok = True)

    torch.save(model.state_dict(), header.model_file_path_model)
    torch.save(data_loader_train, header.model_file_path_data_loader_train)
    torch.save(data_loader_validation, header.model_file_path_data_loader_validation)
    torch.save(epoch, header.model_file_path_epoch)
    torch.save(criterion, header.model_file_path_criterion)
    torch.save(optimizer.state_dict(), header.model_file_path_optimizer)
    torch.save(learning_rate_scheduler.state_dict(), header.model_file_path_learning_rate_scheduler)
    torch.save(accuracy_validation, header.model_file_path_accuracy_validation)

    logger.log_info("Saved training states to \"" + header.model_dir + "\".")

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