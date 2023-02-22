import logger
import math
import torch
import tqdm
import utility
import wandb

def trainBaseline(model, data_loader, criterion, optimizer, device, batch_step):
    accuracy_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    for (i, (input, labels)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)

            if wandb.config.use_l2_loss:
                l2_norm = utility.computeL2Norm(model.parameters())
                loss_l2 = wandb.config.l2_lambda * l2_norm
                loss += loss_l2

            loss.backward()
            optimizer.step()

        corrects = torch.sum(predictions == labels.data).item()

        accuracy_batch = corrects / input.size(0)
        loss_batch = loss.item()

        accuracy_epoch += corrects
        loss_epoch += loss_batch

        progress_bar.n = i + 1
        progress_bar.refresh()

        wandb.log({"training/batch/accuracy": accuracy_batch})
        wandb.log({"training/batch/step": batch_step})
        wandb.log({"training/batch/loss": loss_batch})

        batch_step += 1

    accuracy_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    progress_bar.close()

    wandb.log({"training/epoch/accuracy": accuracy_epoch})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def trainDecomposed(model, dataset, data_loader, epoch, criterions, optimizer, device):
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

                if wandb.config.use_l2_loss:
                    l2_norm = utility.computeL2Norm(model.parameters())
                    loss_l2 = wandb.config.l2_lambda * l2_norm
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
