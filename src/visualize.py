import cv2
import header
import numpy
import os
import torch
import utility

def annotateInput(input, batch_index, config_dataset_generation, dataset_decomposed, ground_truths_label_original, ground_truths_label_decomposed, predictions_label_baseline, predictions_label_composed, predictions_label_decomposed, predictions_confidence_baseline, predictions_confidence_composed, predictions_confidence_decomposed):
    text_position_y = header.visualize_text_position_y_start

    for (task_index, dataset_entry) in enumerate(dataset_decomposed.config["datasets"]):
        input = cv2.putText(input, "Decomposed Prediction for \"" + dataset_entry["name"] + "\": " + predictions_label_decomposed[task_index][batch_index] + " " + str(round(predictions_confidence_decomposed[task_index][batch_index].item() * 100, 2)) + "% ", (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, header.visualize_text_scale, (0, 0, 255), header.visualize_text_thickness, cv2.LINE_AA)
        text_position_y += header.visualize_text_position_y_increment
        input = cv2.putText(input, "Ground Truth for \"" + dataset_entry["name"] + "\": " + ground_truths_label_decomposed[task_index][batch_index], (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, header.visualize_text_scale, (0, 255, 0), header.visualize_text_thickness, cv2.LINE_AA)
        text_position_y += header.visualize_text_position_y_increment
        input = cv2.putText(input, "Weight for \"" + dataset_entry["name"] + "\": " + str(round(config_dataset_generation[ground_truths_label_original[batch_index]]["weights"][dataset_entry["name"]], 2)), (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, header.visualize_text_scale, (255, 100, 100), header.visualize_text_thickness, cv2.LINE_AA)
        text_position_y += header.visualize_text_position_y_increment
        text_position_y += header.visualize_text_position_y_increment

    input = cv2.putText(input, "Composed Prediction: " + predictions_label_composed[batch_index] + " " + str(round(predictions_confidence_composed[batch_index].item() * 100, 2)) + "% ", (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, header.visualize_text_scale, (0, 0, 255), header.visualize_text_thickness, cv2.LINE_AA)
    text_position_y += header.visualize_text_position_y_increment

    input = cv2.putText(input, "Baseline Prediction: " + predictions_label_baseline[batch_index] + " " + str(round(predictions_confidence_baseline[batch_index].item() * 100, 2)) + "% ", (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, header.visualize_text_scale, (0, 0, 255), header.visualize_text_thickness, cv2.LINE_AA)
    text_position_y += header.visualize_text_position_y_increment

    input = cv2.putText(input, "Ground Truth: " + ground_truths_label_original[batch_index], (20, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, header.visualize_text_scale, (0, 255, 0), header.visualize_text_thickness, cv2.LINE_AA)
    text_position_y += header.visualize_text_position_y_increment
    text_position_y += header.visualize_text_position_y_increment

    return input

def gatherDecomposedConfidence(dataset_decomposed, outputs_decomposed, predictions_label_decomposed):
    predictions_confidence_decomposed = []

    for (task_index, output_decomposed) in enumerate(outputs_decomposed):
        prediction_confidence_decomposed_batch = []

        for (batch_index, output_decomposed_batch) in enumerate(output_decomposed):
            prediction_label_decomposed = predictions_label_decomposed[task_index][batch_index]
            prediction_index_decomposed_task = dataset_decomposed.class_to_idx[task_index][prediction_label_decomposed]

            prediction_confidence_decomposed_batch.append(output_decomposed_batch[prediction_index_decomposed_task])

        predictions_confidence_decomposed.append(prediction_confidence_decomposed_batch)

    return predictions_confidence_decomposed

def indicesToLabels(config_dataset_generation, dataset_decomposed, ground_truths_index_decomposed, ground_truths_index_original, predictions_index_baseline, predictions_index_composed):
    ground_truths_label_original = []
    ground_truths_label_decomposed = []
    predictions_label_baseline = []
    predictions_label_composed = []
    predictions_label_decomposed = []

    for prediction_index_baseline in predictions_index_baseline:
        predictions_label_baseline.append(dataset_decomposed.classes_original[prediction_index_baseline])

    for prediction_index_composed in predictions_index_composed:
        predictions_label_composed.append(dataset_decomposed.classes_original[prediction_index_composed])

    for ground_truth_index_original in ground_truths_index_original:
        ground_truths_label_original.append(dataset_decomposed.classes_original[ground_truth_index_original])

    for _ in dataset_decomposed.config["datasets"]:
        ground_truths_label_decomposed.append([])

    for ground_truth_index_decomposed_batch in ground_truths_index_decomposed:
        for (task_index, ground_truth_index_decomposed) in enumerate(ground_truth_index_decomposed_batch):
            ground_truths_label_decomposed[task_index].append(dataset_decomposed.classes[task_index][ground_truth_index_decomposed])

    for (task_index, dataset) in enumerate(dataset_decomposed.config["datasets"]):
        prediction_label_decomposed_batch = []

        for prediction_label_composed in predictions_label_composed:
            prediction_label_decomposed = config_dataset_generation[prediction_label_composed]["labels"][dataset["name"]]

            if prediction_label_decomposed == "":
                prediction_label_decomposed = dataset["name"] + header.config_decomposed["dataset_delimiter_label"] + header.config_decomposed["dataset_label_undefined_keyword"]

            prediction_label_decomposed_batch.append(prediction_label_decomposed)

        predictions_label_decomposed.append(prediction_label_decomposed_batch)

    return (ground_truths_label_original, ground_truths_label_decomposed, predictions_label_baseline, predictions_label_composed, predictions_label_decomposed)

def saveImage(image, dir_output, ground_truth_label):
    if not hasattr(saveImage, "file_count_map"):
        saveImage.file_count_map = {}

    if ground_truth_label not in saveImage.file_count_map:
        saveImage.file_count_map[ground_truth_label] = 0

    dir_image = os.path.join(dir_output, ground_truth_label)

    if not os.path.isdir(dir_image):
        os.makedirs(dir_image, exist_ok = True)

    file_path_image = os.path.join(dir_image, ground_truth_label + "_" + str(saveImage.file_count_map[ground_truth_label]) + ".png")
    cv2.imwrite(file_path_image, image)
    saveImage.file_count_map[ground_truth_label] += 1

    return

def transformInput(input):
    input = numpy.transpose(input, (1, 2, 0))
    input = numpy.clip(input, 0, 1)
    input = input.astype(numpy.float32)
    input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
    input *= 255.0
    input = input.astype(numpy.uint8)
    input = utility.resize(input, height = header.visualize_size)

    return input

def visualize(input, config_dataset_generation, dataset_decomposed, ground_truths_index_decomposed, ground_truths_index_original, output_baseline, outputs_decomposed, output_composed):
    softmax = torch.nn.Softmax(dim = 1)
    input = input.cpu().numpy()
    output_baseline = softmax(output_baseline)

    (predictions_confidence_baseline, predictions_index_baseline) = torch.max(output_baseline, 1)
    (predictions_confidence_composed, predictions_index_composed) = torch.max(output_composed, 1)
    (ground_truths_label_original, ground_truths_label_decomposed, predictions_label_baseline, predictions_label_composed, predictions_label_decomposed) = indicesToLabels(config_dataset_generation, dataset_decomposed, ground_truths_index_decomposed, ground_truths_index_original, predictions_index_baseline, predictions_index_composed)
    predictions_confidence_decomposed = gatherDecomposedConfidence(dataset_decomposed, outputs_decomposed, predictions_label_decomposed)

    if header.visualize_show:
        cv2.namedWindow(dataset_decomposed.root, cv2.WINDOW_NORMAL)

    for (batch_index, input_batch) in enumerate(input):
        input_batch = transformInput(input_batch)
        input_batch = annotateInput(input_batch, batch_index, config_dataset_generation, dataset_decomposed, ground_truths_label_original, ground_truths_label_decomposed, predictions_label_baseline, predictions_label_composed, predictions_label_decomposed, predictions_confidence_baseline, predictions_confidence_composed, predictions_confidence_decomposed)

        if header.visualize_show:
            cv2.imshow(dataset_decomposed.root, input_batch)
            cv2.waitKey(0)

        if header.visualize_save:
            if predictions_label_composed[batch_index] == ground_truths_label_original[batch_index]:
                saveImage(input_batch, header.visualize_dir_output_correct, ground_truths_label_original[batch_index])
            else:
                saveImage(input_batch, header.visualize_dir_output_incorrect, ground_truths_label_original[batch_index])

    if header.visualize_show:
        cv2.destroyAllWindows()

    return
