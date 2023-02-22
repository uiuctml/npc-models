
import datetime
import json
import logger
import os
import socket
import torch
import wandb

def computeL2Norm(parameters):
    parameters_list = []

    for parameter in parameters:
        parameters_list.append(parameter.view(-1))

    return torch.square(torch.cat(parameters_list)).sum().item()

def generateRunName(model_name):
    date_time_list = list(datetime.datetime.now().timetuple())[:-4]
    run_name = model_name

    for entry in date_time_list:
        run_name += "."
        run_name += str(entry)

    run_name += "."
    run_name += socket.gethostname()

    return run_name

def loadEvaluation(model_dir):
    classes = None
    class_indices_test = None
    outputs_test = None

    if os.path.isdir(model_dir):
        model_file_path_class_indices_test = os.path.join(model_dir, wandb.config.model_file_name_class_indices_test)
        model_file_path_classes = os.path.join(model_dir, wandb.config.model_file_name_classes)
        model_file_path_outputs_test = os.path.join(model_dir, wandb.config.model_file_name_outputs_test)

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
        model_file_path_model_best = os.path.join(model_dir, wandb.config.model_file_name_model_best)

        if os.path.isfile(model_file_path_model_best) and model != None:
            model.load_state_dict(torch.load(model_file_path_model_best))
            logger.log_info("Loaded best training model state from \"" + model_dir + "\".")

    return

def loadTraining(model_dir, model, optimizer, learning_rate_scheduler):
    accuracy_validation = None
    criterion = None
    data_loader_test = None
    data_loader_train = None
    data_loader_validation = None
    epoch = None
    statistics_train = None

    if os.path.isdir(model_dir):
        model_file_path_accuracy_validation = os.path.join(model_dir, wandb.config.model_file_name_accuracy_validation)
        model_file_path_criterion = os.path.join(model_dir, wandb.config.model_file_name_criterion)
        model_file_path_data_loader_test = os.path.join(model_dir, wandb.config.model_file_name_data_loader_test)
        model_file_path_data_loader_train = os.path.join(model_dir, wandb.config.model_file_name_data_loader_train)
        model_file_path_data_loader_validation = os.path.join(model_dir, wandb.config.model_file_name_data_loader_validation)
        model_file_path_epoch = os.path.join(model_dir, wandb.config.model_file_name_epoch)
        model_file_path_learning_rate_scheduler = os.path.join(model_dir, wandb.config.model_file_name_learning_rate_scheduler)
        model_file_path_model = os.path.join(model_dir, wandb.config.model_file_name_model)
        model_file_path_optimizer = os.path.join(model_dir, wandb.config.model_file_name_optimizer)
        model_file_path_statistics_train = os.path.join(model_dir, wandb.config.model_file_name_statistics_train)

        if os.path.isfile(model_file_path_model) and model != None:
            model.load_state_dict(torch.load(model_file_path_model))
            logger.log_info("Loaded training model state from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_data_loader_test):
            data_loader_test = torch.load(model_file_path_data_loader_test)
            logger.log_info("Loaded training test data loader from \"" + model_dir + "\".")

        if os.path.isfile(model_file_path_data_loader_train):
            data_loader_train = torch.load(model_file_path_data_loader_train)
            logger.log_info("Loaded training train data loader from \"" + model_dir + "\".")

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

    return (data_loader_test, data_loader_train, data_loader_validation, epoch, criterion, accuracy_validation, statistics_train)

def loadTrainingBest(model_dir, model):
    if os.path.isdir(model_dir):
        model_file_path_model_best = os.path.join(model_dir, wandb.config.model_file_name_model_best)

        if os.path.isfile(model_file_path_model_best) and model != None:
            model_state_dict = torch.load(model_file_path_model_best)

            del model_state_dict["module.fc.1.weight"]
            del model_state_dict["module.fc.1.bias"]

            model.load_state_dict(model_state_dict, strict = False)
            logger.log_info("Loaded best training model state from \"" + model_dir + "\".")

    return

def saveTesting(model_dir, outputs, class_indices, classes, statistics):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok = True)

    model_file_path_class_indices_test = os.path.join(model_dir, wandb.config.model_file_name_class_indices_test)
    model_file_path_classes = os.path.join(model_dir, wandb.config.model_file_name_classes)
    model_file_path_outputs_test = os.path.join(model_dir, wandb.config.model_file_name_outputs_test)
    model_file_path_statistics_test = os.path.join(model_dir, wandb.config.model_file_name_statistics_test)

    torch.save(class_indices, model_file_path_class_indices_test)
    torch.save(classes, model_file_path_classes)
    torch.save(outputs, model_file_path_outputs_test)

    with open(model_file_path_statistics_test, "w") as file_statistics_test:
        json.dump(statistics, file_statistics_test, indent = 4)

    logger.log_info("Saved testing data and statistics to \"" + model_dir + "\".")

    return

def saveTraining(model_dir, model, data_loader_test, data_loader_train, data_loader_validation, epoch, criterion, optimizer, learning_rate_scheduler, accuracy_validation, statistics, best):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok = True)

    model_file_path_accuracy_validation = os.path.join(model_dir, wandb.config.model_file_name_accuracy_validation)
    model_file_path_criterion = os.path.join(model_dir, wandb.config.model_file_name_criterion)
    model_file_path_data_loader_test = os.path.join(model_dir, wandb.config.model_file_name_data_loader_test)
    model_file_path_data_loader_train = os.path.join(model_dir, wandb.config.model_file_name_data_loader_train)
    model_file_path_data_loader_validation = os.path.join(model_dir, wandb.config.model_file_name_data_loader_validation)
    model_file_path_epoch = os.path.join(model_dir, wandb.config.model_file_name_epoch)
    model_file_path_learning_rate_scheduler = os.path.join(model_dir, wandb.config.model_file_name_learning_rate_scheduler)
    model_file_path_model = os.path.join(model_dir, wandb.config.model_file_name_model)
    model_file_path_model_best = os.path.join(model_dir, wandb.config.model_file_name_model_best)
    model_file_path_optimizer = os.path.join(model_dir, wandb.config.model_file_name_optimizer)
    model_file_path_statistics_train = os.path.join(model_dir, wandb.config.model_file_name_statistics_train)

    if best:
        torch.save(model.state_dict(), model_file_path_model_best)

    torch.save(model.state_dict(), model_file_path_model)
    torch.save(data_loader_test, model_file_path_data_loader_test)
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
