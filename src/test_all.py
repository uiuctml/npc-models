#!/usr/bin/env python3

import composition
import dataset
import gc
import header
import logger
import network
import numpy
import sklearn.metrics
import spn
import torch
import torch.nn
import tqdm
import utility
import visualize

def main():
    utility.processArgumentsTest()

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    accuracy_epoch_baseline = 0
    accuracy_epoch_composed = 0
    accuracy_epoch_list_decomposed = []
    ground_truths_epoch_baseline = []
    ground_truths_epoch_list_decomposed = []
    output_list_baseline = []
    output_list_composed = []
    output_list_decomposed = []
    output_list_decomposed_accuracy = []
    output_list_decomposed_precision = []
    output_list_decomposed_recall = []
    predictions_epoch_baseline = []
    predictions_epoch_composed = []
    predictions_epoch_list_decomposed = []
    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_baseline = network.createModelBaseline(device)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)
    model_decomposed = network.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    spn_joint = spn.SPN()
    spn_marginal = spn.SPN()

    logger.log_info("Loading SPN from \"" + header.file_path_spn + "\"...")

    spn_joint.load(header.file_path_spn)
    spn_marginal.load(header.file_path_spn)

    logger.log_info("Loading SPN leaf node settings...")

    spn_settings_joint = numpy.load(header.file_path_spn_settings).astype(numpy.int32)
    spn_settings_marginal = numpy.copy(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    spn_settings_joint = torch.Tensor(spn_settings_joint)
    spn_settings_joint = spn_settings_joint.to(device)
    spn_settings_marginal = torch.Tensor(spn_settings_marginal)
    spn_settings_marginal = spn_settings_marginal.to(device)

    gc.collect()

    logger.log_info("SPN leaf node setting dimension:", int(spn_settings_joint.shape[0]), int(spn_settings_joint.shape[1]))

    spn_output_rows = len(dataset_test.classes_original)
    spn_output_cols = 1

    for attribute in dataset_test.classes:
        spn_output_cols *= len(attribute)

    logger.log_info("SPN output dimension:", int(spn_output_rows), int(spn_output_cols))

    logger.log_info("Setting SPN leaf nodes...")

    spn_joint.set_leaf_nodes(spn_settings_joint)
    spn_marginal.set_leaf_nodes(spn_settings_marginal)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list_decomposed.append(0)
        ground_truths_epoch_list_decomposed.append([])
        predictions_epoch_list_decomposed.append([])

    utility.loadCheckpointBest(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint_best"], model_baseline)
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)

    progress_bar = tqdm.tqdm(total = len(data_loader_test), position = 0, leave = False)

    model_baseline.eval()
    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Inference progress")

    with torch.no_grad():
        for (batch_index, (input, labels_decomposed, labels_original)) in enumerate(data_loader_test):
            input = input.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output_baseline = model_baseline(input)
                (outputs_decomposed, _) = model_decomposed(input)

                output_baseline = utility.applySoftmax(output_baseline)
                outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed)

                output_composed = composition.Composition.spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)

                (_, predictions_baseline) = torch.max(output_baseline, 1)
                (_, predictions_composed) = torch.max(output_composed, 1)

                corrects_baseline = torch.sum(predictions_baseline == labels_original.data).item()
                corrects_composed = torch.sum(predictions_composed == labels_original.data).item()

                for i in range(0, len(config_dataset["attributes"])):
                    (_, predictions_decomposed) = torch.max(outputs_decomposed[i], 1)

                    corrects_decomposed = torch.sum(predictions_decomposed == labels_decomposed[:, i].data).item()
                    accuracy_epoch_list_decomposed[i] += corrects_decomposed
                    ground_truths_epoch_list_decomposed[i] += labels_decomposed[:, i].data.tolist()
                    predictions_epoch_list_decomposed[i] += predictions_decomposed.tolist()

            accuracy_epoch_baseline += corrects_baseline
            accuracy_epoch_composed += corrects_composed

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            ground_truths_epoch_baseline += labels_original.data.tolist()
            predictions_epoch_baseline += predictions_baseline.tolist()
            predictions_epoch_composed += predictions_composed.tolist()

            if header.visualize_show or header.visualize_save:
                visualize.visualize(input, composition, dataset_test, labels_decomposed, labels_original, output_baseline, outputs_decomposed, output_composed)

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list_decomposed[i] /= len(data_loader_test.dataset)
        precision_epoch_decomposed = sklearn.metrics.precision_score(ground_truths_epoch_list_decomposed[i], predictions_epoch_list_decomposed[i], average = "macro", zero_division = 0)
        recall_epoch_decomposed = sklearn.metrics.recall_score(ground_truths_epoch_list_decomposed[i], predictions_epoch_list_decomposed[i], average = "macro", zero_division = 0)

        output_list_decomposed_accuracy.append(accuracy_epoch_list_decomposed[i])
        output_list_decomposed_precision.append(precision_epoch_decomposed)
        output_list_decomposed_recall.append(recall_epoch_decomposed)

        logger.log_info("Decomposed testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_epoch_list_decomposed[i]) + ".")

    accuracy_epoch_baseline /= len(data_loader_test.dataset)
    accuracy_epoch_composed /= len(data_loader_test.dataset)

    precision_epoch_baseline = sklearn.metrics.precision_score(ground_truths_epoch_baseline, predictions_epoch_baseline, average = "macro", zero_division = 0)
    precision_epoch_composed = sklearn.metrics.precision_score(ground_truths_epoch_baseline, predictions_epoch_composed, average = "macro", zero_division = 0)
    recall_epoch_baseline = sklearn.metrics.recall_score(ground_truths_epoch_baseline, predictions_epoch_baseline, average = "macro", zero_division = 0)
    recall_epoch_composed = sklearn.metrics.recall_score(ground_truths_epoch_baseline, predictions_epoch_composed, average = "macro", zero_division = 0)
    output_list_baseline += [accuracy_epoch_baseline, precision_epoch_baseline, recall_epoch_baseline]
    output_list_composed += [accuracy_epoch_composed, precision_epoch_composed, recall_epoch_composed]
    output_list_decomposed += output_list_decomposed_accuracy
    output_list_decomposed += output_list_decomposed_precision
    output_list_decomposed += output_list_decomposed_recall

    logger.log_info("Baseline testing accuracy: " + str(accuracy_epoch_baseline) + ".")
    logger.log_info("Composed testing accuracy: " + str(accuracy_epoch_composed) + ".")

    utility.logTestOutput(header.config_baseline, output_list_baseline)
    utility.logTestOutput(header.config_baseline, output_list_composed, True)
    utility.logTestOutput(header.config_decomposed, output_list_decomposed)

    return

if __name__ == "__main__":
    main()
