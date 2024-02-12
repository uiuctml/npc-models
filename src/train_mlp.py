#!/usr/bin/env python3

import dataset
import header
import logger
import math
import network
import sklearn.metrics
import torch
import torch.nn
import torch.optim
import torchsummary
import tqdm
import utility
import wandb

def test(model, config_dataset, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model)

    accuracy_epoch_list = []
    ground_truths_epoch_list = []
    output_list = []
    output_list_accuracy = []
    output_list_precision = []
    output_list_recall = []
    predictions_epoch_list = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list.append(0)
        ground_truths_epoch_list.append([])
        predictions_epoch_list.append([])

    model.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.no_grad():
        for (batch_index, (input, labels, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                outputs = model(input)

                for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
                    (_, predictions) = torch.max(outputs[i], 1)

                    corrects = torch.sum(predictions == labels[:, i].data).item()
                    accuracy_batch = corrects / input.size(0)
                    accuracy_epoch_list[i] += corrects

                    wandb.log({"testing/batch/" + dataset_entry["name"] + "/accuracy": accuracy_batch})

                    ground_truths_epoch_list[i] += labels[:, i].data.tolist()
                    predictions_epoch_list[i] += predictions.tolist()

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        precision_epoch = sklearn.metrics.precision_score(ground_truths_epoch_list[i], predictions_epoch_list[i], average = "macro", zero_division = 0)
        recall_epoch = sklearn.metrics.recall_score(ground_truths_epoch_list[i], predictions_epoch_list[i], average = "macro", zero_division = 0)

        output_list_accuracy.append(accuracy_epoch_list[i])
        output_list_precision.append(precision_epoch)
        output_list_recall.append(recall_epoch)

        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/precision": precision_epoch})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/recall": recall_epoch})
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/accuracy"] = accuracy_epoch_list[i]
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/precision"] = precision_epoch
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/recall"] = recall_epoch

        logger.log_info("Testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_epoch_list[i]) + ".")
        logger.log_trace("Testing precision for \"" + dataset_entry["name"] + "\": " + str(precision_epoch) + ".")
        logger.log_trace("Testing recall for \"" + dataset_entry["name"] + "\": " + str(recall_epoch) + ".")

    output_list += output_list_accuracy
    output_list += output_list_precision
    output_list += output_list_recall

    utility.logTestOutput(header.config_decomposed, output_list)

    return batch_step

def train(model, config_dataset, data_loader, criterions, optimizers, device, batch_step):
    accuracy_epoch_list = []
    loss_epoch_list = []
    loss_overall_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list.append(0)
        loss_epoch_list.append(0)

    model.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    for (batch_index, (input, labels, _)) in enumerate(data_loader):
        outputs = model(input)

        with torch.set_grad_enabled(True):
            parameters = model.getOptimizerParameters()

            for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
                optimizer = optimizers[i]
                optimizer.zero_grad()

                input = input.to(device, non_blocking = True)
                labels = labels.to(device, non_blocking = True)

                (_, predictions) = torch.max(outputs[i], 1)
                loss = criterions[i](outputs[i], labels[:, i])

                if header.config_decomposed["use_l2_loss"]:
                    l2_norm = utility.computeL2Norm(parameters[i])
                    loss_l2 = header.config_decomposed["l2_lambda"] * l2_norm
                    loss += loss_l2

                loss.backward()
                optimizer.step()

                corrects = torch.sum(predictions == labels[:, i].data).item()

                accuracy_batch = corrects / input.size(0)
                loss_batch = loss.item()

                accuracy_epoch_list[i] += corrects
                loss_epoch_list[i] += loss_batch

                wandb.log({"training/batch/" + dataset_entry["name"] + "/accuracy": accuracy_batch})
                wandb.log({"training/batch/" + dataset_entry["name"] + "/loss": loss_batch})

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

        wandb.log({"training/batch/step": batch_step})

        batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        loss_epoch_list[i] /= len(data_loader)
        wandb.log({"training/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"training/epoch/" + dataset_entry["name"] + "/loss": loss_epoch_list[i]})

    loss_overall_epoch /= len(data_loader)

    wandb.log({"training/epoch/loss": loss_overall_epoch})

    return batch_step

def validate(model, config_dataset, data_loader, criterions, device, batch_step):
    accuracy_epoch_list = []
    loss_epoch_list = []
    loss_overall_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list.append(0)
        loss_epoch_list.append(0)

    model.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    for (batch_index, (input, labels, _)) in enumerate(data_loader):
        loss_overall = 0

        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            outputs = model(input)

            for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
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
                loss_overall += loss / math.log(outputs[i].size(1))

                wandb.log({"validation/batch/" + dataset_entry["name"] + "/accuracy": accuracy_batch})
                wandb.log({"validation/batch/" + dataset_entry["name"] + "/loss": loss_batch})

            loss_overall /= len(outputs)

        loss_overall_batch = loss_overall.item()
        loss_overall_epoch += loss_overall_batch

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

        wandb.log({"validation/batch/step": batch_step})
        wandb.log({"validation/batch/loss": loss_overall_batch})

        batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        loss_epoch_list[i] /= len(data_loader)
        wandb.log({"validation/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"validation/epoch/" + dataset_entry["name"] + "/loss": loss_epoch_list[i]})

    loss_overall_epoch /= len(data_loader)

    wandb.log({"validation/epoch/loss": loss_overall_epoch})

    return (accuracy_epoch_list, loss_overall_epoch, batch_step)

def main():
    resume = utility.processArgumentsTrainDecomposed()

    utility.setSeed(header.config_decomposed["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.run_name_decomposed, config = header.config_decomposed, resume = resume, mode = header.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + header.run_name_decomposed + "\".")

    accuracy_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterions = []
    optimizers = []
    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.VISATDataset(header.config_decomposed["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.VISATDataset(header.config_decomposed["dir_dataset_validation"], dataset_transforms)
    config_dataset = dataset_train.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model = network.createModelDecomposed(device)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    progress_bar = None

    parameters = model.module.getOptimizerParameters()
    print(parameters)

    for i in range(len(config_dataset["attributes"])):
        optimizer = torch.optim.SGD(parameters[i], lr = header.config_decomposed["optimizer_learning_rate"], momentum = header.config_decomposed["optimizer_momentum"], weight_decay = header.config_decomposed["optimizer_weight_decay"])

        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.config_decomposed["learning_rate_scheduler_mode"], header.config_decomposed["learning_rate_scheduler_factor"], header.config_decomposed["learning_rate_scheduler_patience"], header.config_decomposed["learning_rate_scheduler_threshold"], header.config_decomposed["learning_rate_scheduler_threshold_mode"], header.config_decomposed["learning_rate_scheduler_cooldown"], header.config_decomposed["learning_rate_scheduler_min_learning_rate"], header.config_decomposed["learning_rate_scheduler_min_learning_rate_decay"], header.config_decomposed["learning_rate_scheduler_verbose"])

        optimizers.append(optimizer)
        criterions.append(torch.nn.CrossEntropyLoss())

    (accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch) = utility.loadCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_scheduler, model, optimizer)

    if header.show_model_summary:
        model_input_size = (header.config_decomposed["model_input_channels"], header.config_decomposed["model_input_height"], header.config_decomposed["model_input_width"])
        torchsummary.summary(model, input_size = model_input_size)

    if epoch <= header.config_decomposed["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_decomposed["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_decomposed["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model, config_dataset, data_loader_train, criterions, optimizers, device, batch_step_train)
        (accuracy_validation_epoch_list, loss_overall_validation_epoch, batch_step_validate) = validate(model, config_dataset, data_loader_validation, criterions, device, batch_step_validate)

        learning_rate_scheduler.step(loss_overall_validation_epoch)

        accuracy_validation_epoch_mean = sum(accuracy_validation_epoch_list) / len(accuracy_validation_epoch_list)

        if accuracy_validation_epoch_mean > accuracy_validation_best:
            accuracy_validation_best = accuracy_validation_epoch_mean
            wandb.log({"validation/epoch/accuracy_best": accuracy_validation_best})
            utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_scheduler, model, optimizer)

        utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_scheduler, model, optimizer)

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation accuracy: " + str(accuracy_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_best"] = accuracy_validation_best

    wandb.log({"testing/epoch/step": 1})
    batch_step_test = test(model, config_dataset, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
