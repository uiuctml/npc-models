
import header
import json
import logger
import math
import matplotlib.pyplot
import numpy
import os
import torch

def computeL2Norm(parameters):
    parameters_list = []

    for parameter in parameters:
        parameters_list.append(parameter.view(-1))

    return torch.square(torch.cat(parameters_list)).sum().item()

def loadEvaluation(model_dir):
    classes = None
    class_indices_test = None
    outputs_test = None

    if os.path.isdir(model_dir):
        model_file_path_class_indices_test = os.path.join(model_dir, header.model_file_name_class_indices_test)
        model_file_path_classes = os.path.join(model_dir, header.model_file_name_classes)
        model_file_path_outputs_test = os.path.join(model_dir, header.model_file_name_outputs_test)

        if os.path.isfile(model_file_path_class_indices_test):
            class_indices_test = torch.load(model_file_path_class_indices_test)
            logger.log_info("Loaded testing class indices from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_classes):
            classes = torch.load(model_file_path_classes)
            logger.log_info("Loaded classes from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_outputs_test):
            outputs_test = torch.load(model_file_path_outputs_test)
            logger.log_info("Loaded testing outputs from \"" + model_dir + "\".")

    return (outputs_test, class_indices_test, classes)

def loadTesting(model_dir, model):
    if os.path.isdir(model_dir):
        model_file_path_model_best = os.path.join(model_dir, header.model_file_name_model_best)

        if os.path.isfile(model_file_path_model_best) and model != None:
            model.load_state_dict(torch.load(model_file_path_model_best))
            logger.log_info("Loaded best training model state from \"" + model_dir + "\".")

    return

def loadTraining(model_dir, model, optimizer, learning_rate_scheduler):
    accuracy_validation = None
    criterion = None
    data_loader_train = None
    data_loader_validation = None
    epoch = None
    statistics_train = None

    if os.path.isdir(model_dir):
        model_file_path_accuracy_validation = os.path.join(model_dir, header.model_file_name_accuracy_validation)
        model_file_path_criterion = os.path.join(model_dir, header.model_file_name_criterion)
        model_file_path_data_loader_train = os.path.join(model_dir, header.model_file_name_data_loader_train)
        model_file_path_data_loader_validation = os.path.join(model_dir, header.model_file_name_data_loader_validation)
        model_file_path_epoch = os.path.join(model_dir, header.model_file_name_epoch)
        model_file_path_learning_rate_scheduler = os.path.join(model_dir, header.model_file_name_learning_rate_scheduler)
        model_file_path_model = os.path.join(model_dir, header.model_file_name_model)
        model_file_path_optimizer = os.path.join(model_dir, header.model_file_name_optimizer)
        model_file_path_statistics_train = os.path.join(model_dir, header.model_file_name_statistics_train)

        if os.path.isfile(model_file_path_model) and model != None:
            model.load_state_dict(torch.load(model_file_path_model))
            logger.log_info("Loaded training model state from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_data_loader_train):
            data_loader_train = torch.load(model_file_path_data_loader_train)
            logger.log_info("Loaded training data loader from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_data_loader_validation):
            data_loader_validation = torch.load(model_file_path_data_loader_validation)
            logger.log_info("Loaded training validation data loader from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_epoch):
            epoch = torch.load(model_file_path_epoch)
            logger.log_info("Loaded training epoch from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_criterion):
            criterion = torch.load(model_file_path_criterion)
            logger.log_info("Loaded training criterion from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_optimizer) and optimizer != None:
            optimizer.load_state_dict(torch.load(model_file_path_optimizer))
            logger.log_info("Loaded training optimizer state from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_learning_rate_scheduler) and learning_rate_scheduler != None:
            learning_rate_scheduler.load_state_dict(torch.load(model_file_path_learning_rate_scheduler))
            logger.log_info("Loaded training learning rate scheduler state from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_accuracy_validation):
            accuracy_validation = torch.load(model_file_path_accuracy_validation)
            logger.log_info("Loaded training highest accuracy validation from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_statistics_train):
            with open(model_file_path_statistics_train, "r") as file_statistics_train:
                statistics_train = json.load(file_statistics_train)
                logger.log_info("Loaded training statistics from \"" + model_dir + "\".")

    return (data_loader_train, data_loader_validation, epoch, criterion, accuracy_validation, statistics_train)

def loadTrainingBest(model_dir, model):
    if os.path.isdir(model_dir):
        model_file_path_model_best = os.path.join(model_dir, header.model_file_name_model_best)

        if os.path.isfile(model_file_path_model_best) and model != None:
            model_state_dict = torch.load(model_file_path_model_best)

            del model_state_dict["module.fc.1.weight"]
            del model_state_dict["module.fc.1.bias"]

            model.load_state_dict(model_state_dict, strict = False)
            logger.log_info("Loaded best training model state from \"" + model_dir + "\".")

    return

def plotEvaluationStatistics(statistics):
    if statistics == None:
        return

    average_precisions = numpy.array(list(statistics.values()))
    average_precisions_indices = range(0, len(average_precisions))
    average_precisions_split = numpy.array_split(average_precisions, header.plot_subplot_count_evaluate)
    average_precisions_indices_split = numpy.array_split(average_precisions_indices, header.plot_subplot_count_evaluate)

    figure = matplotlib.pyplot.figure()
    figure_manager = matplotlib.pyplot.get_current_fig_manager()
    subplot_rows = math.ceil(header.plot_subplot_count_evaluate / header.plot_subplot_col_count_evaluate)
    subplot_cols = header.plot_subplot_col_count_evaluate

    figure.suptitle("Evaluation Statistics for \"" + header.plot_model_dir + "\"")

    for i in range(0, header.plot_subplot_count_evaluate):
        figure.add_subplot(subplot_rows, subplot_cols, i + 1)
        matplotlib.pyplot.bar(average_precisions_indices_split[i], average_precisions_split[i])
        matplotlib.pyplot.xlabel("Class")
        matplotlib.pyplot.ylabel("Average Precision")

    figure_manager.full_screen_toggle()

    if header.plot_show_evaluate:
        matplotlib.pyplot.show()

    if header.plot_save_evaluate:
        figure.savefig(header.plot_model_dir.split("/")[-1] + "_evaluate.png")

    return

def plotTestingStatistics(statistics):
    if statistics == None:
        return

    accuracies = statistics["testing_accuracies"]
    accuracy_mean = numpy.nanmean(accuracies)

    figure = matplotlib.pyplot.figure()
    figure_manager = matplotlib.pyplot.get_current_fig_manager()

    matplotlib.pyplot.plot(accuracies)
    matplotlib.pyplot.axhline(y = accuracy_mean, color = "red")
    matplotlib.pyplot.title("Testing Statistics for \"" + header.plot_model_dir + "\"")
    matplotlib.pyplot.xlabel("Batch")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim([0, 1])
    matplotlib.pyplot.yticks(list(matplotlib.pyplot.yticks()[0]) + [accuracy_mean])

    figure_manager.full_screen_toggle()

    if header.plot_show_test:
        matplotlib.pyplot.show()

    if header.plot_save_test:
        figure.savefig(header.plot_model_dir.split("/")[-1] + "_test.png")

    return

def plotTrainingStatistics(statistics):
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

    figure = matplotlib.pyplot.figure()
    figure_manager = matplotlib.pyplot.get_current_fig_manager()

    figure.suptitle("Training Statistics for \"" + header.plot_model_dir + "\"")

    figure.add_subplot(2, 2, 1)
    matplotlib.pyplot.plot(accuracies_training)
    matplotlib.pyplot.xticks(accuracies_training_xticks, range(0, len(accuracies_training_xticks)))
    matplotlib.pyplot.title("Training Accuracy")
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim([0, 1])

    figure.add_subplot(2, 2, 2)
    matplotlib.pyplot.plot(accuracies_validation)
    matplotlib.pyplot.xticks(accuracies_validation_xticks, range(0, len(accuracies_validation_xticks)))
    matplotlib.pyplot.title("Validation Accuracy")
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim([0, 1])

    figure.add_subplot(2, 2, 3)
    matplotlib.pyplot.plot(losses_training)
    matplotlib.pyplot.xticks(losses_training_xticks, range(0, len(losses_training_xticks)))
    matplotlib.pyplot.title("Training Loss")
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Loss")

    figure.add_subplot(2, 2, 4)
    matplotlib.pyplot.plot(losses_validation)
    matplotlib.pyplot.xticks(losses_validation_xticks, range(0, len(losses_validation_xticks)))
    matplotlib.pyplot.title("Validation Loss")
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Loss")

    figure_manager.full_screen_toggle()

    if header.plot_show_train:
        matplotlib.pyplot.show()

    if header.plot_save_train:
        figure.savefig(header.plot_model_dir.split("/")[-1] + "_train.png")

    return

def saveEvaluation(model_dir, statistics):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok = True)

    model_file_path_statistics_evaluate = os.path.join(model_dir, header.model_file_name_statistics_evaluate)

    with open(model_file_path_statistics_evaluate, "w") as file_statistics_evaluate:
        json.dump(statistics, file_statistics_evaluate, indent = 4)

    logger.log_info("Saved evaluation statistics to \"" + model_dir + "\".")

    return

def saveTesting(model_dir, outputs, class_indices, classes, statistics):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok = True)

    model_file_path_class_indices_test = os.path.join(model_dir, header.model_file_name_class_indices_test)
    model_file_path_classes = os.path.join(model_dir, header.model_file_name_classes)
    model_file_path_outputs_test = os.path.join(model_dir, header.model_file_name_outputs_test)
    model_file_path_statistics_test = os.path.join(model_dir, header.model_file_name_statistics_test)

    torch.save(class_indices, model_file_path_class_indices_test)
    torch.save(classes, model_file_path_classes)
    torch.save(outputs, model_file_path_outputs_test)

    with open(model_file_path_statistics_test, "w") as file_statistics_test:
        json.dump(statistics, file_statistics_test, indent = 4)

    logger.log_info("Saved testing data and statistics to \"" + model_dir + "\".")

    return

def saveTraining(model_dir, model, data_loader_train, data_loader_validation, epoch, criterion, optimizer, learning_rate_scheduler, accuracy_validation, statistics, best):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok = True)

    model_file_path_accuracy_validation = os.path.join(model_dir, header.model_file_name_accuracy_validation)
    model_file_path_criterion = os.path.join(model_dir, header.model_file_name_criterion)
    model_file_path_data_loader_train = os.path.join(model_dir, header.model_file_name_data_loader_train)
    model_file_path_data_loader_validation = os.path.join(model_dir, header.model_file_name_data_loader_validation)
    model_file_path_epoch = os.path.join(model_dir, header.model_file_name_epoch)
    model_file_path_learning_rate_scheduler = os.path.join(model_dir, header.model_file_name_learning_rate_scheduler)
    model_file_path_model = os.path.join(model_dir, header.model_file_name_model)
    model_file_path_model_best = os.path.join(model_dir, header.model_file_name_model_best)
    model_file_path_optimizer = os.path.join(model_dir, header.model_file_name_optimizer)
    model_file_path_statistics_train = os.path.join(model_dir, header.model_file_name_statistics_train)

    if best:
        torch.save(model.state_dict(), model_file_path_model_best)

    torch.save(model.state_dict(), model_file_path_model)
    torch.save(data_loader_train, model_file_path_data_loader_train)
    torch.save(data_loader_validation, model_file_path_data_loader_validation)
    torch.save(epoch, model_file_path_epoch)
    torch.save(criterion, model_file_path_criterion)
    torch.save(optimizer.state_dict(), model_file_path_optimizer)
    torch.save(learning_rate_scheduler.state_dict(), model_file_path_learning_rate_scheduler)
    torch.save(accuracy_validation, model_file_path_accuracy_validation)

    with open(model_file_path_statistics_train, "w") as file_statistics_train:
        json.dump(statistics, file_statistics_train, indent = 4)

    logger.log_info("Saved training states and statistics to \"" + model_dir + "\".")

    return

def viewDataset(dataset, data_loader):
    if header.log_level < logger.LogLevel.trace:
        return

    figure_rows = header.dataset_view_row_count
    figure_cols = math.ceil(header.train_data_loader_batch_size / header.dataset_view_row_count)
    figure = matplotlib.pyplot.figure()
    figure_manager = matplotlib.pyplot.get_current_fig_manager()
    input, labels = next(iter(data_loader))
    input = input.numpy().transpose((0, 2, 3, 1))
    class_list = list(dataset.classes[label] for label in labels)

    for i in range(0, header.train_data_loader_batch_size):
        figure.add_subplot(figure_cols, figure_rows, i + 1)
        matplotlib.pyplot.title(class_list[i])
        matplotlib.pyplot.axis("off")
        matplotlib.pyplot.imshow(input[i].squeeze())

    figure_manager.full_screen_toggle()
    matplotlib.pyplot.show()

    return