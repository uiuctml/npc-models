import logger
import torch
import tqdm
import utility
import wandb
import sklearn.metrics

def testBaseline(model, data_loader, device, batch_step):
    accuracy_epoch = 0
    ground_truths_epoch = []
    predictions_epoch = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    utility.loadCheckpointBest(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, model)

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
    logger.log_info("Testing precision: " + str(precision_epoch) + ".")
    logger.log_info("Testing recall: " + str(recall_epoch) + ".")

    return batch_step

def testDecomposed(model, dataset, device, batch_step):
    data_loader = utility.loadCheckpointBest(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, model)

    if data_loader is None:
        logger.log_error("Data loader missing.")
        return batch_step

    accuracy_epoch_list = []
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    ground_truths_epoch_list = []
    predictions_epoch_list = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    for _ in dataset.config["datasets"]:
        accuracy_epoch_list.append(0)
        ground_truths_epoch_list.append([])
        predictions_epoch_list.append([])

    model.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.no_grad():
        for (batch_index, (input, labels)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                outputs = model(input)

                for (i, dataset_entry) in enumerate(dataset.config["datasets"]):
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

    for (i, dataset_entry) in enumerate(dataset.config["datasets"]):
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
        logger.log_info("Testing precision for \"" + dataset_entry["name"] + "\": " + str(precision_epoch) + ".")
        logger.log_info("Testing recall for \"" + dataset_entry["name"] + "\": " + str(recall_epoch) + ".")

    return batch_step
