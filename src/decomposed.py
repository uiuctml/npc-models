#!/usr/bin/env python3

import dataset
import header
import logger
import math
import network
import torch
import torch.nn
import torch.optim
import torchsummary
import torchvision
import tqdm
import type
import utility

def test(model, data_loader, device, statistics):
    accuracies = []
    batch_count = math.ceil(len(data_loader.dataset) / header.decomposed_data_loader_batch_size)
    class_indices = []
    running_corrects = 0
    outputs = []
    progress_bar_accuracy = tqdm.tqdm(total = 1, position = 1, leave = False)
    progress_bar_progress = tqdm.tqdm(total = batch_count, position = 0, leave = False)
    progress_bar_accuracy.set_description_str("[INFO]: Testing accuracy")
    progress_bar_progress.set_description_str("[INFO]: Testing progress")

    model.eval()

    with torch.no_grad():
        for (i, (input, labels)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output = model(input)
                (_, predictions) = torch.max(output, 1)

            corrects = torch.sum(predictions == labels.data).item()
            accuracy_value = corrects / input.size(0)

            running_corrects += corrects

            progress_bar_accuracy.n = round(accuracy_value, 4)
            progress_bar_progress.n = i + 1

            progress_bar_accuracy.refresh()
            progress_bar_progress.refresh()

            accuracies.append(accuracy_value)
            class_indices.append(labels.data.tolist())
            outputs.append(output.tolist())

    statistics["testing_accuracies"] = accuracies

    progress_bar_accuracy.close()
    progress_bar_progress.close()

    return (running_corrects, outputs, class_indices)

def train(model, dataset, data_loader, epoch, criterions, optimizer, device, statistics):
    accuracies = []
    losses = []
    running_loss = 0
    running_corrects_list = []
    progress_bar_accuracy_list = []
    progress_bar_accuracy_position = 3 + len(dataset.config["datasets"])
    progress_bar_loss_list = []
    progress_bar_loss_position = 3
    progress_bar_loss_overall = tqdm.tqdm(total = 10, position = 2, leave = False)
    progress_bar_progress = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)
    progress_bar_loss_overall.set_description_str("[INFO]: Training overall loss")
    progress_bar_progress.set_description_str("[INFO]: Training progress")

    for dataset_entry in dataset.config["datasets"]:
        progress_bar_accuracy = tqdm.tqdm(total = 1, position = progress_bar_accuracy_position, leave = False)
        progress_bar_loss = tqdm.tqdm(total = 10, position = progress_bar_loss_position, leave = False)
        progress_bar_accuracy.set_description_str("[INFO]: Training accuracy for \"" + dataset_entry["name"] + "\"")
        progress_bar_loss.set_description_str("[INFO]: Training loss for \"" + dataset_entry["name"] + "\"")

        running_corrects_list.append(0)
        progress_bar_accuracy_list.append(progress_bar_accuracy)
        progress_bar_loss_list.append(progress_bar_loss)

        progress_bar_accuracy_position += 1
        progress_bar_loss_position += 1

    model.train()

    for (batch_index, (input, labels)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        loss_overall = None

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            loss_list = []
            predictions_list = []

            outputs = model(input)

            for i in range(0, len(dataset.config["datasets"])):
                logger.log_trace("outputs[i]:", outputs[i])
                logger.log_trace("labels[:, i]:", labels[:, i])
                logger.log_trace("outputs[i].shape:", outputs[i].shape)
                logger.log_trace("labels[:, i].shape:", labels[:, i].shape)

                (_, predictions) = torch.max(outputs[i], 1)
                loss = criterions[i](outputs[i], labels[:, i])

                if header.decomposed_use_l2_loss:
                    l2_norm = utility.computeL2Norm(model.parameters())
                    loss_l2 = header.decomposed_l2_lambda * l2_norm
                    loss += loss_l2

                loss_list.append(loss)
                predictions_list.append(predictions)

            for (i, loss) in enumerate(loss_list):
                if loss_overall == None:
                    loss_overall = loss / math.log(outputs[i].size(1))
                else:
                    loss_overall += loss / math.log(outputs[i].size(1))

            logger.log_trace("len(outputs):", len(outputs))

            loss_overall /= len(outputs)

            loss_overall.backward()
            optimizer.step()

        accuracy_value_list = []

        for i in range(0, len(dataset.config["datasets"])):
            corrects = torch.sum(predictions_list[i] == labels[:, i].data).item()
            accuracy_value = corrects / input.size(0)

            progress_bar_accuracy_list[i].n = round(accuracy_value, 4)
            progress_bar_loss_list[i].n = round(loss_list[i].item(), 4)

            progress_bar_accuracy_list[i].refresh()
            progress_bar_loss_list[i].refresh()

            accuracy_value_list.append(accuracy_value)
            running_corrects_list[i] += corrects

        loss_overall_value = loss_overall.item()
        running_loss += loss_overall_value
        
        progress_bar_loss_overall.n = round(loss_overall_value, 4)
        progress_bar_progress.n = batch_index + 1

        progress_bar_loss_overall.refresh()
        progress_bar_progress.refresh()

        accuracies.append(accuracy_value_list)
        losses.append(loss_overall_value)

    for i in range(0, len(dataset.config["datasets"])):
        progress_bar_accuracy_list[i].close()
        progress_bar_loss_list[i].close()

    progress_bar_loss_overall.close()
    progress_bar_progress.close()

    statistics["epoch_" + str(epoch)]["training_accuracies"] = accuracies
    statistics["epoch_" + str(epoch)]["training_losses"] = losses

    return (running_loss, running_corrects_list)

def validate(model, dataset, data_loader, epoch, criterions, device, statistics):
    accuracies = []
    losses = []
    running_loss = 0
    running_corrects_list = []
    progress_bar_accuracy_list = []
    progress_bar_accuracy_position = 3 + len(dataset.config["datasets"])
    progress_bar_loss_list = []
    progress_bar_loss_position = 3
    progress_bar_loss_overall = tqdm.tqdm(total = 10, position = 2, leave = False)
    progress_bar_progress = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)
    progress_bar_loss_overall.set_description_str("[INFO]: Validation overall loss")
    progress_bar_progress.set_description_str("[INFO]: Validation progress")

    for dataset_entry in dataset.config["datasets"]:
        progress_bar_accuracy = tqdm.tqdm(total = 1, position = progress_bar_accuracy_position, leave = False)
        progress_bar_loss = tqdm.tqdm(total = 10, position = progress_bar_loss_position, leave = False)
        progress_bar_accuracy.set_description_str("[INFO]: Validation accuracy for \"" + dataset_entry["name"] + "\"")
        progress_bar_loss.set_description_str("[INFO]: Validation loss for \"" + dataset_entry["name"] + "\"")

        running_corrects_list.append(0)
        progress_bar_accuracy_list.append(progress_bar_accuracy)
        progress_bar_loss_list.append(progress_bar_loss)

        progress_bar_accuracy_position += 1
        progress_bar_loss_position += 1

    model.eval()

    for (batch_index, (input, labels)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        loss_overall = None

        with torch.set_grad_enabled(False):
            loss_list = []
            predictions_list = []

            outputs = model(input)

            for i in range(0, len(dataset.config["datasets"])):
                (_, predictions) = torch.max(outputs[i], 1)
                loss = criterions[i](outputs[i], labels[:, i])

                loss_list.append(loss)
                predictions_list.append(predictions)

            for (i, loss) in enumerate(loss_list):
                if loss_overall == None:
                    loss_overall = loss / math.log(outputs[i].size(1))
                else:
                    loss_overall += loss / math.log(outputs[i].size(1))

            loss_overall /= len(outputs)

        accuracy_value_list = []

        for i in range(0, len(dataset.config["datasets"])):
            corrects = torch.sum(predictions_list[i] == labels[:, i].data).item()
            accuracy_value = corrects / input.size(0)

            progress_bar_accuracy_list[i].n = round(accuracy_value, 4)
            progress_bar_loss_list[i].n = round(loss_list[i].item(), 4)

            progress_bar_accuracy_list[i].refresh()
            progress_bar_loss_list[i].refresh()

            accuracy_value_list.append(accuracy_value)
            running_corrects_list[i] += corrects

        loss_overall_value = loss_overall.item()
        running_loss += loss_overall_value

        progress_bar_loss_overall.n = round(loss_overall_value, 4)
        progress_bar_progress.n = batch_index + 1

        progress_bar_loss_overall.refresh()
        progress_bar_progress.refresh()

        accuracies.append(accuracy_value_list)
        losses.append(loss_overall_value)

    for i in range(0, len(dataset.config["datasets"])):
        progress_bar_accuracy_list[i].close()
        progress_bar_loss_list[i].close()

    progress_bar_loss_overall.close()
    progress_bar_progress.close()

    statistics["epoch_" + str(epoch)]["validation_accuracies"] = accuracies
    statistics["epoch_" + str(epoch)]["validation_losses"] = losses

    return (running_loss, running_corrects_list)

def main():
    accuracy_validation = None
    criterions = None
    data_loader_test = None
    data_loader_train = None
    data_loader_validation = None
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.decomposed_model_input_height, header.decomposed_model_input_width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    device = torch.device("cuda")
    epoch = None
    progress_bar = None
    statistics_test = {}
    statistics_train = None

    dataset_generated = dataset.DatasetGenerated(root = header.dataset_dir_generated, transform = dataset_transforms)

    model = network.DecomposedNetworkA(dataset_generated)

    if header.decomposed_load_best:
        utility.loadTrainingBest(header.decomposed_model_dir_best, model)

    if not header.decomposed_fine_tuning:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = header.decomposed_optimizer_learning_rate, momentum = header.decomposed_optimizer_momentum, weight_decay = header.decomposed_optimizer_weight_decay)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.decomposed_learning_rate_scheduler_mode, header.decomposed_learning_rate_scheduler_factor, header.decomposed_learning_rate_scheduler_patient, header.decomposed_learning_rate_scheduler_threshold, header.decomposed_learning_rate_scheduler_threshold_mode, header.decomposed_learning_rate_scheduler_cooldown, header.decomposed_learning_rate_scheduler_min_learning_rate, header.decomposed_learning_rate_scheduler_min_learning_rate_decay, header.decomposed_learning_rate_scheduler_verbose)

    if not header.decomposed_dry_run:
        (data_loader_test, data_loader_train, data_loader_validation, epoch, criterions, accuracy_validation, statistics_train) = utility.loadTraining(header.decomposed_model_dir, model, optimizer, learning_rate_scheduler)
        logger.log_info_raw("\n")

    if accuracy_validation == None:
        accuracy_validation = 0

    if criterions == None:
        criterions = []

        for _ in range(0, len(dataset_generated.config["datasets"])):
            criterions.append(torch.nn.CrossEntropyLoss())

    if data_loader_train == None or data_loader_validation == None or data_loader_test == None:
        dataset_split_lengths = [header.decomposed_dataset_split_percentage_test, header.decomposed_dataset_split_percentage_train, header.decomposed_dataset_split_percentage_validation]
        (dataset_subset_test, dataset_subset_train, dataset_subset_validation) = torch.utils.data.random_split(dataset_generated, dataset_split_lengths)

        data_loader_test = torch.utils.data.DataLoader(dataset_subset_test, batch_size = header.decomposed_data_loader_batch_size, shuffle = header.decomposed_data_loader_shuffle, num_workers = header.decomposed_data_loader_worker_count, pin_memory = True)
        data_loader_train = torch.utils.data.DataLoader(dataset_subset_train, batch_size = header.decomposed_data_loader_batch_size, shuffle = header.decomposed_data_loader_shuffle, num_workers = header.decomposed_data_loader_worker_count, pin_memory = True)
        data_loader_validation = torch.utils.data.DataLoader(dataset_subset_validation, batch_size = header.decomposed_data_loader_batch_size, shuffle = header.decomposed_data_loader_shuffle, num_workers = header.decomposed_data_loader_worker_count, pin_memory = True)

    if epoch == None:
        epoch = 1

    if statistics_train == None:
        statistics_train = {}

    if header.log_level >= type.LogLevel.debug:
        model_input_size = (header.decomposed_model_input_channels, header.decomposed_model_input_height, header.decomposed_model_input_width)
        torchsummary.summary(model, input_size = model_input_size)

    utility.viewDatasetDecomposed(dataset_generated, data_loader_train, header.decomposed_data_loader_batch_size)
    utility.viewDatasetDecomposed(dataset_generated, data_loader_validation, header.decomposed_data_loader_batch_size)
    utility.viewDatasetDecomposed(dataset_generated, data_loader_test, header.decomposed_data_loader_batch_size)

    if epoch <= header.decomposed_epochs:
        progress_bar = tqdm.tqdm(total = header.decomposed_epochs, position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")
        progress_bar.n = epoch
        progress_bar.refresh()

    while epoch <= header.decomposed_epochs:
        best = False
        statistics_epoch_train = (0, 0)
        statistics_epoch_validation = (0, 0)

        if not header.decomposed_dry_run:
            statistics_train["epoch_" + str(epoch)] = {}
            statistics_epoch_train = train(model, dataset_generated, data_loader_train, epoch, criterions, optimizer, device, statistics_train)
            statistics_epoch_validation = validate(model, dataset_generated, data_loader_validation, epoch, criterions, device, statistics_train)

        epoch_mean_accuracy_validation = 0
        epoch_loss_train = statistics_epoch_train[0] / len(data_loader_train)
        epoch_loss_validation = statistics_epoch_validation[0] / len(data_loader_validation)

        logger.log_info("Training loss: " + str(epoch_loss_train) + ".")

        for (i, dataset_entry) in enumerate(dataset_generated.config["datasets"]):
            epoch_accuracy_train = statistics_epoch_train[1][i] / len(data_loader_train.dataset)
            logger.log_info("Training accuracy for \"" + dataset_entry["name"] + "\": " + str(epoch_accuracy_train) + ".")

        logger.log_info("Validation loss: " + str(epoch_loss_validation) + ".")

        for (i, dataset_entry) in enumerate(dataset_generated.config["datasets"]):
            epoch_accuracy_validation = statistics_epoch_validation[1][i] / len(data_loader_validation.dataset)
            epoch_mean_accuracy_validation += epoch_accuracy_validation
            logger.log_info("Validation accuracy for \"" + dataset_entry["name"] + "\": " + str(epoch_accuracy_validation) + ".")

        learning_rate_scheduler.step(epoch_loss_validation)
        epoch += 1
        epoch_mean_accuracy_validation /= len(dataset_generated.config["datasets"])

        if epoch_mean_accuracy_validation > accuracy_validation:
            accuracy_validation = epoch_mean_accuracy_validation
            best = True

        if not header.decomposed_dry_run:
            utility.saveTraining(header.decomposed_model_dir, model, data_loader_test, data_loader_train, data_loader_validation, epoch, criterions, optimizer, learning_rate_scheduler, accuracy_validation, statistics_train, best)

        logger.log_info_raw("\n")

        if progress_bar != None and epoch <= header.decomposed_epochs:
            progress_bar.n = epoch
            progress_bar.refresh()

    if progress_bar != None:
        progress_bar.close()

    logger.log_info("Highest validation accuracy: " + str(accuracy_validation) + ".")
    logger.log_info_raw("\n")

    statistics_epoch_test = (0, [], [])

    utility.loadTesting(header.decomposed_model_dir, model)

    logger.log_info("Testing best model in \"" + header.decomposed_model_dir + "\".")

    if not header.decomposed_dry_run:
        statistics_epoch_test = test(model, data_loader_test, device, statistics_test)

    accuracy_test = statistics_epoch_test[0] / len(data_loader_test.dataset)

    logger.log_info("Testing accuracy: " + str(accuracy_test) + ".")

    if not header.decomposed_dry_run:
        utility.saveTesting(header.decomposed_model_dir, statistics_epoch_test[1], statistics_epoch_test[2], dataset_test.classes, statistics_test)

    return

if __name__ == "__main__":
    main()
