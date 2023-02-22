import logger
import torch
import tqdm
import utility
import wandb

def testBaseline(model, dataset, data_loader, device, batch_step):
    utility.loadCheckpointBest(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, model)

    accuracy_epoch = 0
    ground_truths_epoch = []
    predictions_epoch = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    model.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.no_grad():
        for (i, (input, labels)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output = model(input)
                (_, predictions) = torch.max(output, 1)

            corrects = torch.sum(predictions == labels.data).item()

            accuracy_batch = corrects / input.size(0)
            accuracy_epoch += corrects

            progress_bar.n = i + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy": accuracy_batch})
            wandb.log({"testing/batch/step": batch_step})

            ground_truths_epoch += labels.data.tolist()
            predictions_epoch += output.tolist()

            batch_step += 1

    accuracy_epoch /= len(data_loader.dataset)

    progress_bar.close()

    pr_curve = wandb.plot.pr_curve(ground_truths_epoch, predictions_epoch, labels = dataset.classes, title = "Precision vs. Recall")
    roc_curve = wandb.plot.roc_curve(ground_truths_epoch, predictions_epoch, labels = dataset.classes, title = "Receiver Operating Characteristic")

    wandb.log({"testing/epoch/accuracy": accuracy_epoch})
    wandb.log({"testing/epoch/pr_curve": pr_curve})
    wandb.log({"testing/epoch/roc_curve": roc_curve})

    logger.log_info("Testing accuracy: " + str(accuracy_epoch) + ".")
    wandb.summary["testing/epoch/accuracy"] = accuracy_epoch

    return batch_step

def testDecomposed(model, dataset, data_loader, device):
    accuracies = []
    class_indices_list = []
    running_corrects_list = []
    outputs_list = []
    progress_bar_accuracy_list = []
    progress_bar_accuracy_position = 1
    progress_bar_progress = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    progress_bar_progress.set_description_str("[INFO]: Testing progress")

    for dataset_entry in dataset.config["datasets"]:
        progress_bar_accuracy = tqdm.tqdm(total = 1, position = progress_bar_accuracy_position, leave = False)
        progress_bar_accuracy.set_description_str("[INFO]: Testing accuracy for \"" + dataset_entry["name"] + "\"")

        class_indices_list.append([])
        outputs_list.append([])
        running_corrects_list.append(0)
        progress_bar_accuracy_list.append(progress_bar_accuracy)

        progress_bar_accuracy_position += 1

    model.eval()

    with torch.no_grad():
        for (batch_index, (input, labels)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                predictions_list = []

                outputs = model(input)

                for i in range(0, len(dataset.config["datasets"])):
                    (_, predictions) = torch.max(outputs[i], 1)
                    predictions_list.append(predictions)

            accuracy_value_list = []

            for i in range(0, len(dataset.config["datasets"])):
                corrects = torch.sum(predictions_list[i] == labels[:, i].data).item()
                accuracy_value = corrects / input.size(0)

                progress_bar_accuracy_list[i].n = round(accuracy_value, 4)
                progress_bar_accuracy_list[i].refresh()

                accuracy_value_list.append(accuracy_value)
                running_corrects_list[i] += corrects

                class_indices_list[i] = labels[:, i].data.tolist()
                outputs_list[i] = outputs[i].tolist()

            progress_bar_progress.n = batch_index + 1
            progress_bar_progress.refresh()

            accuracies.append(accuracy_value_list)

    statistics["testing_accuracies"] = accuracies

    for i in range(0, len(dataset.config["datasets"])):
        progress_bar_accuracy_list[i].close()

    progress_bar_progress.close()

    return (running_corrects_list, outputs_list, class_indices_list)
