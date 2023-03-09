#!/usr/bin/env python3

import composition as comp
import cv2
import dataset as dset
import header
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

def annotateInput(input, output_baseline, output_decomposed, outputs_decomposed, labels_original, labels_decomposed, dataset_decomposed, config_dataset_generation, ground_truth_label_counts):
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
        predictions_label_baseline.append(dataset_decomposed.classes_original[prediction_index_baseline])

    for prediction_index_decomposed in predictions_index_decomposed:
        predictions_label_decomposed.append(dataset_decomposed.classes_original[prediction_index_decomposed])

    for label_original in labels_original:
        ground_truths_label_original.append(dataset_decomposed.classes_original[label_original])

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
                prediction_label_decomposed_task = task_name + header.config_decomposed["dataset_delimiter_label"] + header.config_decomposed["dataset_label_undefined_keyword"]

            prediction_index_decomposed_task = dataset_decomposed.config["datasets"][task_index]["labels"].index(prediction_label_decomposed_task)

            prediction_confidence_decomposed_task_batch.append(output_decomposed_task[prediction_index_decomposed_task])
            prediction_label_decomposed_task_batch.append(prediction_label_decomposed_task)

        predictions_confidence_decomposed_task.append(prediction_confidence_decomposed_task_batch)
        predictions_label_decomposed_task.append(prediction_label_decomposed_task_batch)

    input_cpu = input.cpu().numpy()
    mean = numpy.array([0.5, 0.5, 0.5])
    std = numpy.array([0.5, 0.5, 0.5])

    cv2.namedWindow(header.config_decomposed["dir_dataset_test"], cv2.WINDOW_NORMAL)

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
            cv2.imshow(header.config_decomposed["dir_dataset_test"], input_batch)
            cv2.waitKey(0)
        else:
            save(input_batch, ground_truth_label_counts, ground_truths_label_original[batch_index], correct)

    if show:
        cv2.destroyAllWindows()

    return

def main():
    run_name_baseline = ""
    run_name_decomposed = ""

    if len(sys.argv) > 2:
        run_name_baseline = sys.argv[1]
        run_name_decomposed = sys.argv[2]
    else:
        logger.log_error("Run names missing. Quit.")
        return

    if not utility.initializeRunNameBaseline(run_name_baseline) or header.run_name_baseline == "":
        logger.log_error("Baseline run name missing. Quit.")
        return

    if not utility.initializeRunNameDecomposed(run_name_decomposed) or header.run_name_decomposed == "":
        logger.log_error("Decomposed run name missing. Quit.")
        return

    accuracy_epoch_baseline = 0
    accuracy_epoch_composed = 0
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.config_decomposed["model_input_height"], header.config_decomposed["model_input_width"])),
        torchvision.transforms.ToTensor(),
    ])
    dataset_original = torchvision.datasets.ImageFolder(header.config_baseline["dir_dataset_test"], dataset_transforms)
    dataset_test = dset.DatasetDecomposed(header.config_decomposed["dir_dataset_test"], dataset_original.classes, dataset_transforms)
    config_dataset = dataset_test.config
    class_count_original = len(dataset_original.classes)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    composition = comp.Composition(dataset_test, device)
    ground_truth_label_counts = {}
    model_baseline = network.BaselineNetworkA(class_count_original)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)
    model_decomposed = network.DecomposedNetworkA(config_dataset)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)

    utility.loadCheckpointBest(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint_best"], model_baseline)
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)

    progress_bar = tqdm.tqdm(total = len(data_loader_test), position = 0, leave = False)

    model_baseline.eval()
    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Inference progress")

    with torch.no_grad():
        for (batch_index, (input, labels, labels_original)) in enumerate(data_loader_test):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output_baseline = model_baseline(input)
                outputs_decomposed = model_decomposed(input)
                output_composed = composition.compose(outputs_decomposed)

                annotateInput(input, output_baseline, output_composed, outputs_decomposed, labels_original, labels, dataset_test, composition.config_generate, ground_truth_label_counts)

                (_, predictions_baseline) = torch.max(output_baseline, 1)
                (_, predictions_composed) = torch.max(output_composed, 1)

                corrects_baseline = torch.sum(predictions_baseline == labels_original.data).item()
                corrects_composed = torch.sum(predictions_composed == labels_original.data).item()

            accuracy_epoch_baseline += corrects_baseline
            accuracy_epoch_composed += corrects_composed

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

    progress_bar.close()

    accuracy_epoch_baseline /= len(data_loader_test.dataset)
    accuracy_epoch_composed /= len(data_loader_test.dataset)

    logger.log_info("Baseline inference accuracy: " + str(accuracy_epoch_baseline) + ".")
    logger.log_info("Composed inference accuracy: " + str(accuracy_epoch_composed) + ".")

    return

if __name__ == "__main__":
    main()
