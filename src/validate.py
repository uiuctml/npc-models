import math
import torch
import tqdm

def validateBaseline(model, data_loader, epoch, criterion, device, statistics):
    accuracies = []
    losses = []
    running_loss = 0
    running_corrects = 0
    progress_bar_accuracy = tqdm.tqdm(total = 1, position = 3, leave = False)
    progress_bar_loss = tqdm.tqdm(total = 10, position = 2, leave = False)
    progress_bar_progress = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)
    progress_bar_accuracy.set_description_str("[INFO]: Validation accuracy")
    progress_bar_loss.set_description_str("[INFO]: Validation loss")
    progress_bar_progress.set_description_str("[INFO]: Validation progress")

    model.eval()

    for (i, (input, labels)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            output = model(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)

        corrects = torch.sum(predictions == labels.data).item()
        accuracy_value = corrects / input.size(0)
        loss_value = loss.item()

        running_loss += loss_value
        running_corrects += corrects

        progress_bar_accuracy.n = round(accuracy_value, 4)
        progress_bar_loss.n = round(loss_value, 4)
        progress_bar_progress.n = i + 1

        progress_bar_accuracy.refresh()
        progress_bar_loss.refresh()
        progress_bar_progress.refresh()

        accuracies.append(accuracy_value)
        losses.append(loss_value)

    progress_bar_accuracy.close()
    progress_bar_loss.close()
    progress_bar_progress.close()

    statistics["epoch_" + str(epoch)]["validation_accuracies"] = accuracies
    statistics["epoch_" + str(epoch)]["validation_losses"] = losses

    return (running_loss, running_corrects)

def validateDecomposed(model, dataset, data_loader, epoch, criterions, device, statistics):
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
