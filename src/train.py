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

    for (batch_index, (input, labels)) in enumerate(data_loader):
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

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

        wandb.log({"training/batch/accuracy": accuracy_batch})
        wandb.log({"training/batch/step": batch_step})
        wandb.log({"training/batch/loss": loss_batch})

        batch_step += 1

    progress_bar.close()

    accuracy_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"training/epoch/accuracy": accuracy_epoch})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def trainDecomposed(model, dataset, data_loader, criterions, optimizer, device, batch_step):
    accuracy_epoch_list = []
    loss_epoch_list = []
    loss_overall_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    for _ in dataset.config["datasets"]:
        accuracy_epoch_list.append(0)
        loss_epoch_list.append(0)

    model.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    for (batch_index, (input, labels)) in enumerate(data_loader):
        loss_overall = 0

        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(input)

            for (i, dataset_entry) in enumerate(dataset.config["datasets"]):
                (_, predictions) = torch.max(outputs[i], 1)
                loss = criterions[i](outputs[i], labels[:, i])

                if wandb.config.use_l2_loss:
                    l2_norm = utility.computeL2Norm(model.parameters())
                    loss_l2 = wandb.config.l2_lambda * l2_norm
                    loss += loss_l2

                corrects = torch.sum(predictions == labels[:, i].data).item()

                accuracy_batch = corrects / input.size(0)
                loss_batch = loss.item()

                accuracy_epoch_list[i] += corrects
                loss_epoch_list[i] += loss_batch
                loss_overall += loss / math.log(outputs[i].size(1))

                wandb.log({"training/batch/" + dataset_entry["name"] + "/accuracy": accuracy_batch})
                wandb.log({"training/batch/" + dataset_entry["name"] + "/loss": loss_batch})

            loss_overall /= len(outputs)

            loss_overall.backward()
            optimizer.step()

        loss_overall_batch = loss_overall.item()
        loss_overall_epoch += loss_overall_batch

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

        wandb.log({"training/batch/step": batch_step})

        batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(dataset.config["datasets"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        loss_epoch_list[i] /= len(data_loader)
        wandb.log({"training/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"training/epoch/" + dataset_entry["name"] + "/loss": loss_epoch_list[i]})

    loss_overall_epoch /= len(data_loader)

    wandb.log({"training/epoch/loss_overall": loss_overall_epoch})

    return batch_step
