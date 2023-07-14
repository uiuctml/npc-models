import header
import logger
import torch
import tqdm
import utility
import wandb
import sklearn.metrics

def testBaseline(model, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint_best"], model)

    accuracy_epoch = 0
    ground_truths_epoch = []
    predictions_epoch = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    model.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.no_grad():
        for (batch_index, (input, labels)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output = model(input)
                (_, predictions) = torch.max(output, 1)

            corrects = torch.sum(predictions == labels.data).item()

            accuracy_batch = corrects / input.size(0)
            accuracy_epoch += corrects

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy": accuracy_batch})
            wandb.log({"testing/batch/step": batch_step})

            ground_truths_epoch += labels.data.tolist()
            predictions_epoch += predictions.tolist()

            batch_step += 1

    progress_bar.close()

    accuracy_epoch /= len(data_loader.dataset)
    precision_epoch = sklearn.metrics.precision_score(ground_truths_epoch, predictions_epoch, average = "macro", zero_division = 0)
    recall_epoch = sklearn.metrics.recall_score(ground_truths_epoch, predictions_epoch, average = "macro", zero_division = 0)

    wandb.log({"testing/epoch/accuracy": accuracy_epoch})
    wandb.log({"testing/epoch/precision": precision_epoch})
    wandb.log({"testing/epoch/recall": recall_epoch})
    wandb.summary["testing/epoch/accuracy"] = accuracy_epoch
    wandb.summary["testing/epoch/precision"] = precision_epoch
    wandb.summary["testing/epoch/recall"] = recall_epoch

    logger.log_info("Testing accuracy: " + str(accuracy_epoch) + ".")
    logger.log_trace("Testing precision: " + str(precision_epoch) + ".")
    logger.log_trace("Testing recall: " + str(recall_epoch) + ".")

    return batch_step

def testDecomposed(model, config_dataset, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model)

    accuracy_epoch_list = []
    ground_truths_epoch_list = []
    predictions_epoch_list = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    for _ in config_dataset["datasets"]:
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

                for (i, dataset_entry) in enumerate(config_dataset["datasets"]):
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

    for (i, dataset_entry) in enumerate(config_dataset["datasets"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        precision_epoch = sklearn.metrics.precision_score(ground_truths_epoch_list[i], predictions_epoch_list[i], average = "macro", zero_division = 0)
        recall_epoch = sklearn.metrics.recall_score(ground_truths_epoch_list[i], predictions_epoch_list[i], average = "macro", zero_division = 0)

        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/precision": precision_epoch})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/recall": recall_epoch})
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/accuracy"] = accuracy_epoch_list[i]
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/precision"] = precision_epoch
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/recall"] = recall_epoch

        logger.log_info("Testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_epoch_list[i]) + ".")
        logger.log_trace("Testing precision for \"" + dataset_entry["name"] + "\": " + str(precision_epoch) + ".")
        logger.log_trace("Testing recall for \"" + dataset_entry["name"] + "\": " + str(recall_epoch) + ".")

    return batch_step
