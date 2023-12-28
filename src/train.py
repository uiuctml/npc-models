import header
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

            if header.config_baseline["use_l2_loss"]:
                l2_norm = utility.computeL2Norm(model.parameters())
                loss_l2 = header.config_baseline["l2_lambda"] * l2_norm
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

def trainDecomposed(model, config_dataset, data_loader, criterions, optimizer, device, batch_step):
    accuracy_epoch_list = []
    loss_epoch_list = []
    loss_overall_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    for _ in config_dataset["datasets"]:
        accuracy_epoch_list.append(0)
        loss_epoch_list.append(0)

    model.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    for (batch_index, (input, labels, _)) in enumerate(data_loader):
        loss_covariance = torch.tensor(0)
        loss_criterions = 0

        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            (outputs, outputs_head_hidden) = model(input)

            for (i, dataset_entry) in enumerate(config_dataset["datasets"]):
                (_, predictions) = torch.max(outputs[i], 1)
                loss = criterions[i](outputs[i], labels[:, i])

                if header.config_decomposed["use_l2_loss"]:
                    l2_norm = utility.computeL2Norm(model.parameters())
                    loss_l2 = header.config_decomposed["l2_lambda"] * l2_norm
                    loss += loss_l2

                corrects = torch.sum(predictions == labels[:, i].data).item()

                accuracy_batch = corrects / input.size(0)
                loss_batch = loss.item()

                accuracy_epoch_list[i] += corrects
                loss_epoch_list[i] += loss_batch
                loss_criterions += loss / math.log(outputs[i].size(1))

                wandb.log({"training/batch/" + dataset_entry["name"] + "/accuracy": accuracy_batch})
                wandb.log({"training/batch/" + dataset_entry["name"] + "/loss": loss_batch})

            loss_criterions /= len(outputs)

            if header.config_decomposed["use_covariance_loss"]:
                loss_covariance = utility.computeCovarianceRegularization(outputs_head_hidden)

            loss_overall = loss_criterions + loss_covariance

            loss_overall.backward()
            optimizer.step()

        loss_covariance_batch = loss_covariance.item()
        loss_criterions_batch = loss_criterions.item()
        loss_overall_batch = loss_overall.item()
        loss_overall_epoch += loss_overall_batch

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

        wandb.log({"training/batch/step": batch_step})
        wandb.log({"training/batch/loss": loss_overall_batch})
        wandb.log({"training/batch/loss_covariance": loss_covariance_batch})
        wandb.log({"training/batch/loss_criterions": loss_criterions_batch})

        batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["datasets"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        loss_epoch_list[i] /= len(data_loader)
        wandb.log({"training/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"training/epoch/" + dataset_entry["name"] + "/loss": loss_epoch_list[i]})

    loss_overall_epoch /= len(data_loader)

    wandb.log({"training/epoch/loss": loss_overall_epoch})

    return batch_step
