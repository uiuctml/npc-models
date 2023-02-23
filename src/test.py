import logger
import torch
import tqdm
import utility
import wandb

def testBaseline(model, dataset, data_loader, device, batch_step):
    data_loader = utility.loadCheckpointBest(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, data_loader, model)

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
            predictions_epoch += output.tolist()

            batch_step += 1

    progress_bar.close()

    accuracy_epoch /= len(data_loader.dataset)
    pr_curve = wandb.plot.pr_curve(ground_truths_epoch, predictions_epoch, labels = dataset.classes, title = "Precision vs. Recall")
    roc_curve = wandb.plot.roc_curve(ground_truths_epoch, predictions_epoch, labels = dataset.classes, title = "Receiver Operating Characteristic")

    wandb.log({"testing/epoch/accuracy": accuracy_epoch})
    wandb.log({"testing/epoch/pr_curve": pr_curve})
    wandb.log({"testing/epoch/roc_curve": roc_curve})

    logger.log_info("Testing accuracy: " + str(accuracy_epoch) + ".")
    wandb.summary["testing/epoch/accuracy"] = accuracy_epoch

    return batch_step

def testDecomposed(model, dataset, data_loader, device, batch_step):
    data_loader = utility.loadCheckpointBest(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, data_loader, model)

    accuracy_epoch_list = []
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
                    predictions_epoch_list[i] += outputs[i].tolist()

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(dataset.config["datasets"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        pr_curve = wandb.plot.pr_curve(ground_truths_epoch_list[i], predictions_epoch_list[i], labels = dataset.classes[i], title = "Precision vs. Recall")
        roc_curve = wandb.plot.roc_curve(ground_truths_epoch_list[i], predictions_epoch_list[i], labels = dataset.classes[i], title = "Receiver Operating Characteristic")

        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/pr_curve": pr_curve})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/roc_curve": roc_curve})

        logger.log_info("Testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_epoch_list[i]) + ".")
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/accuracy"] = accuracy_epoch_list[i]

    return batch_step
