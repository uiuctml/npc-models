#!/usr/bin/env python3

import config
import cv2
import dataset as dset
import decision as deci
import logger
import network
import numpy
import os
import sys
import torch
import torch.nn
import torchvision
import tqdm
import utility
import wandb

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    width_image = image.shape[1]
    width_resize = image.shape[1]
    height_image = image.shape[0]
    height_resize = image.shape[0]

    if width is None and height is None:
        return image

    if width is None:
        resize_ratio = height / height_image
        width_resize = int(width_image * resize_ratio)
        height_resize = height
    else:
        resize_ratio = width / width_image
        width_resize = width
        height_resize = int(height_image * resize_ratio)

    return cv2.resize(image, (width_resize, height_resize), interpolation=inter)

def save(image, ground_truth_label_counts, ground_truth_label, correct):
    output_dir = ""

    if ground_truth_label not in ground_truth_label_counts:
        ground_truth_label_counts[ground_truth_label] = 0

    if correct:
        output_dir = os.path.join("output", "correct", ground_truth_label)
    else:
        output_dir = os.path.join("output", "incorrect", ground_truth_label)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok = True)

    output_file_path = os.path.join(output_dir, ground_truth_label + "_" + str(ground_truth_label_counts[ground_truth_label]) + ".png")
    cv2.imwrite(output_file_path, image)

    ground_truth_label_counts[ground_truth_label] += 1

    return

def annotateInput(input, output_baseline, output_decomposed, outputs_decomposed, labels_original, labels_decomposed, classes_original, dataset_decomposed, config_dataset_generation, ground_truth_label_counts):
    show = False
    ground_truths_label_original = []
    ground_truths_label_decomposed_task = []
    predictions_confidence_decomposed_task = []
    predictions_label_baseline = []
    predictions_label_decomposed = []
    predictions_label_decomposed_task = []
    softmax = torch.nn.Softmax(dim = 1)

    for _ in dataset_decomposed.config["datasets"]:
        ground_truths_label_decomposed_task.append([])

    output_baseline = softmax(output_baseline)
    (predictions_confidence_baseline, predictions_index_baseline) = torch.max(output_baseline, 1)
    (predictions_confidence_decomposed, predictions_index_decomposed) = torch.max(output_decomposed, 1)

    for prediction_index_baseline in predictions_index_baseline:
        predictions_label_baseline.append(classes_original[prediction_index_baseline])

    for prediction_index_decomposed in predictions_index_decomposed:
        predictions_label_decomposed.append(classes_original[prediction_index_decomposed])

    for label_original in labels_original:
        ground_truths_label_original.append(classes_original[label_original])

    for label_decomposed_batch in labels_decomposed:
        for (task_index, label_decomposed) in enumerate(label_decomposed_batch):
            ground_truths_label_decomposed_task[task_index].append(dataset_decomposed.classes[task_index][label_decomposed])

    for (task_index, output_decomposed_task_batch) in enumerate(outputs_decomposed):
        prediction_label_decomposed_task_batch = []
        prediction_confidence_decomposed_task_batch = []

        for (batch_index, output_decomposed_task) in enumerate(output_decomposed_task_batch):
            task_name = dataset_decomposed.config["datasets"][task_index]["name"]
            prediction_label_decomposed_original = predictions_label_decomposed[batch_index]
            prediction_label_decomposed_task = config_dataset_generation[prediction_label_decomposed_original]["labels"][task_name]

            if prediction_label_decomposed_task == "":
                prediction_label_decomposed_task = task_name + wandb.config.dataset_delimiter_label + wandb.config.dataset_label_undefined_keyword

            prediction_index_decomposed_task = dataset_decomposed.config["datasets"][task_index]["labels"].index(prediction_label_decomposed_task)

            prediction_confidence_decomposed_task_batch.append(output_decomposed_task[prediction_index_decomposed_task])
            prediction_label_decomposed_task_batch.append(prediction_label_decomposed_task)

        predictions_confidence_decomposed_task.append(prediction_confidence_decomposed_task_batch)
        predictions_label_decomposed_task.append(prediction_label_decomposed_task_batch)

    input_cpu = input.cpu().numpy()
    mean = numpy.array([0.5, 0.5, 0.5])
    std = numpy.array([0.5, 0.5, 0.5])

    cv2.namedWindow(wandb.config.dir_dataset, cv2.WINDOW_NORMAL)

    for (batch_index, input_batch) in enumerate(input_cpu):
        correct = False
        input_batch = numpy.transpose(input_batch, (1, 2, 0))
        input_batch = mean + std * input_batch 
        input_batch = numpy.clip(input_batch, 0, 1)
        input_batch = input_batch.astype(numpy.float32)
        input_batch = cv2.cvtColor(input_batch, cv2.COLOR_RGB2BGR)
        input_batch *= 255.0
        input_batch = input_batch.astype(numpy.uint8)
        input_batch = resize(input_batch, height = 900)
        text_position_y = 30
        text_position_y_increment = 20

        if predictions_label_decomposed[batch_index] == ground_truths_label_original[batch_index]:
            correct = True

        input_batch = cv2.putText(input_batch, "Baseline Prediction: " + str(round(predictions_confidence_baseline[batch_index].item() * 100, 2)) + "% " + predictions_label_baseline[batch_index], (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        text_position_y += text_position_y_increment

        input_batch = cv2.putText(input_batch, "Decomposed Prediction: " + str(round(predictions_confidence_decomposed[batch_index].item() * 100, 2)) + "% " + predictions_label_decomposed[batch_index], (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        text_position_y += text_position_y_increment

        for (task_index, dataset_entry) in enumerate(dataset_decomposed.config["datasets"]):
            input_batch = cv2.putText(input_batch, "Decomposed Prediction for \"" + dataset_entry["name"] + "\": " + str(round(predictions_confidence_decomposed_task[task_index][batch_index].item() * 100, 2)) + "% " + predictions_label_decomposed_task[task_index][batch_index], (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            text_position_y += text_position_y_increment

        input_batch = cv2.putText(input_batch, "Ground Truth: " + ground_truths_label_original[batch_index], (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        text_position_y += text_position_y_increment

        for (task_index, dataset_entry) in enumerate(dataset_decomposed.config["datasets"]):
            input_batch = cv2.putText(input_batch, "Ground Truth for \"" + dataset_entry["name"] + "\": " + ground_truths_label_decomposed_task[task_index][batch_index], (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            text_position_y += text_position_y_increment

        for (task_index, dataset_entry) in enumerate(dataset_decomposed.config["datasets"]):
            input_batch = cv2.putText(input_batch, "Weight for \"" + dataset_entry["name"] + "\": " + str(round(config_dataset_generation[ground_truths_label_original[batch_index]]["weights"][dataset_entry["name"]], 2)), (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2, cv2.LINE_AA)
            text_position_y += text_position_y_increment

        if show:
            cv2.imshow(wandb.config.dir_dataset, input_batch)
            cv2.waitKey(0)
        else:
            save(input_batch, ground_truth_label_counts, ground_truths_label_original[batch_index], correct)

    if show:
        cv2.destroyAllWindows()

    return

def main():
    if len(sys.argv) > 2:
        if sys.argv[1].split(".")[0] != config.run_name_baseline_keyword:
            logger.log_error("Invalid baseline run name. Quit.")
            return

        if sys.argv[2].split(".")[0] != config.run_name_decomposed_keyword:
            logger.log_error("Invalid decomposed run name. Quit.")
            return

        config.run_name_baseline = sys.argv[1]
        config.run_name_decomposed = sys.argv[2]

        config.config_baseline["file_name_checkpoint"] = config.run_name_baseline + ".tar"
        config.config_baseline["file_name_checkpoint_best"] = config.run_name_baseline + ".best.tar"
        config.config_decomposed["file_name_checkpoint"] = config.run_name_decomposed + ".tar"
        config.config_decomposed["file_name_checkpoint_best"] = config.run_name_decomposed + ".best.tar"
    else:
        logger.log_error("Run names missing. Quit.")
        return

    device = torch.device("cuda")
    ground_truth_label_counts = {}

    wandb.init(config = config.config_baseline, mode = "disabled")

    accuracy_epoch_baseline = 0
    dataset_baseline = torchvision.datasets.ImageFolder(root = wandb.config.dir_dataset)
    model_baseline = network.BaselineNetworkA(dataset_baseline)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)

    utility.loadCheckpointBest(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, model_baseline)

    wandb.finish()
    wandb.init(config = config.config_decomposed, mode = "disabled")

    accuracy_epoch_decomposed = 0
    dataset_decomposed = dset.DatasetGenerated(root = wandb.config.dir_dataset)
    decision = deci.Decision(dataset_decomposed, device)
    model_decomposed = network.DecomposedNetworkA(dataset_decomposed)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)

    data_loader = utility.loadCheckpointBest(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, model_decomposed)

    if data_loader is None:
        logger.log_error("Data loader missing.")
        return

    data_loader.dataset.dataset.constructOriginalKeyLabelMap()

    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    model_baseline.eval()
    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Inference progress")

    with torch.no_grad():
        for (batch_index, (input, labels, labels_original)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output_baseline = model_baseline(input)
                outputs_decomposed = model_decomposed(input)
                (output_decomposed, _) = decision.make(outputs_decomposed, labels, input.size(0))

                annotateInput(input, output_baseline, output_decomposed, outputs_decomposed, labels_original, labels, decision.classes, data_loader.dataset.dataset, decision.config, ground_truth_label_counts)

                (_, predictions_baseline) = torch.max(output_baseline, 1)
                (_, predictions_decomposed) = torch.max(output_decomposed, 1)

                corrects_baseline = torch.sum(predictions_baseline == labels_original.data).item()
                corrects_decomposed = torch.sum(predictions_decomposed == labels_original.data).item()

            accuracy_epoch_baseline += corrects_baseline
            accuracy_epoch_decomposed += corrects_decomposed

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

    progress_bar.close()

    accuracy_epoch_baseline /= len(data_loader.dataset)
    accuracy_epoch_decomposed /= len(data_loader.dataset)

    logger.log_info("Testing accuracy for baseline network: " + str(accuracy_epoch_baseline) + ".")
    logger.log_info("Testing accuracy for decomposed network: " + str(accuracy_epoch_decomposed) + ".")

    return

if __name__ == "__main__":
    main()
