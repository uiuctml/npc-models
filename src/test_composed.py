#!/usr/bin/env python3

import composition
import dataset
import header
import logger
import model
import sklearn.metrics
import spn
import torch
import tqdm
import utility

def generateSPNSettings(config_dataset, device):
    attribute_ranges = []
    labels_attribute = utility.getLabelsAttribute(config_dataset)
    labels_original = utility.getLabelsOriginal(config_dataset)

    for attribute in labels_attribute.keys():
        attribute_count = len(labels_attribute[attribute])
        attribute_range = torch.Tensor(range(attribute_count))
        attribute_range = attribute_range.to(device)
        attribute_ranges.append(attribute_range)

        logger.log_trace("Number of attribute labels for \"" + attribute + "\": " + str(attribute_count) + ".")

    original_count = len(labels_original)
    original_range = torch.Tensor(range(original_count))
    original_range = original_range.to(device)

    logger.log_trace("Number of original labels: " + str(original_count) + ".")

    spn_settings = torch.cartesian_prod(*attribute_ranges)
    original_range = original_range.repeat_interleave(spn_settings.shape[0]).reshape(-1, 1)
    spn_settings = spn_settings.repeat(original_count, 1)
    spn_settings = torch.cat((spn_settings, original_range), 1)

    return spn_settings

def test(model_decomposed, spn_joint, spn_marginal, dataset_test, config_dataset, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)
    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])

    accuracy_epoch_composed = 0
    accuracy_epoch_list_decomposed = []
    ground_truths_epoch_composed = []
    ground_truths_epoch_list_decomposed = []
    output_list_composed = []
    output_list_decomposed = []
    output_list_decomposed_accuracy = []
    output_list_decomposed_precision = []
    output_list_decomposed_recall = []
    predictions_epoch_composed = []
    predictions_epoch_list_decomposed = []
    spn_output_rows = len(dataset_test.classes_original)
    spn_output_cols = 1

    for attribute in dataset_test.classes:
        spn_output_cols *= len(attribute)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list_decomposed.append(0)
        ground_truths_epoch_list_decomposed.append([])
        predictions_epoch_list_decomposed.append([])

    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Inference progress")

    with torch.no_grad():
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                (outputs_decomposed, _) = model_decomposed(input)

                outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed)
                output_composed = composition.Composition.spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)

                (_, predictions_composed) = torch.max(output_composed, 1)

                corrects_composed = torch.sum(predictions_composed == labels_original.data).item()

                for i in range(0, len(config_dataset["attributes"])):
                    (_, predictions_decomposed) = torch.max(outputs_decomposed[i], 1)

                    corrects_decomposed = torch.sum(predictions_decomposed == labels_decomposed[:, i].data).item()
                    accuracy_epoch_list_decomposed[i] += corrects_decomposed
                    ground_truths_epoch_list_decomposed[i] += labels_decomposed[:, i].data.tolist()
                    predictions_epoch_list_decomposed[i] += predictions_decomposed.tolist()

            accuracy_epoch_composed += corrects_composed
            ground_truths_epoch_composed += labels_original.data.tolist()
            predictions_epoch_composed += predictions_composed.tolist()

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list_decomposed[i] /= len(data_loader.dataset)
        precision_epoch_decomposed = sklearn.metrics.precision_score(ground_truths_epoch_list_decomposed[i], predictions_epoch_list_decomposed[i], average = "macro", zero_division = 0)
        recall_epoch_decomposed = sklearn.metrics.recall_score(ground_truths_epoch_list_decomposed[i], predictions_epoch_list_decomposed[i], average = "macro", zero_division = 0)

        output_list_decomposed_accuracy.append(accuracy_epoch_list_decomposed[i])
        output_list_decomposed_precision.append(precision_epoch_decomposed)
        output_list_decomposed_recall.append(recall_epoch_decomposed)

        logger.log_info("Decomposed testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_epoch_list_decomposed[i]) + ".")

    accuracy_epoch_composed /= len(data_loader.dataset)

    precision_epoch_composed = sklearn.metrics.precision_score(ground_truths_epoch_composed, predictions_epoch_composed, average = "macro", zero_division = 0)
    recall_epoch_composed = sklearn.metrics.recall_score(ground_truths_epoch_composed, predictions_epoch_composed, average = "macro", zero_division = 0)
    output_list_composed += [accuracy_epoch_composed, precision_epoch_composed, recall_epoch_composed]
    output_list_decomposed += output_list_decomposed_accuracy
    output_list_decomposed += output_list_decomposed_precision
    output_list_decomposed += output_list_decomposed_recall

    logger.log_info("Composed testing accuracy: " + str(accuracy_epoch_composed) + ".")

    utility.logTestOutput(output_list_composed, header.config_baseline, config_dataset, True)
    utility.logTestOutput(output_list_decomposed, header.config_decomposed, config_dataset)

    return

def main():
    utility.processArgumentsTestComposed()

    header.run_name_baseline = header.config_decomposed["run_name"]
    header.config_baseline["dir_dataset_test"] = header.config_decomposed["dir_dataset_test"]
    header.config_baseline["file_name_checkpoint"] = header.run_name_decomposed + ".tar"
    header.config_baseline["file_name_checkpoint_best"] = header.run_name_decomposed + ".best.tar"
    header.config_baseline["run_name"] = header.run_name_decomposed

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    spn_joint = spn.SPN()
    spn_marginal = spn.SPN()

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])

    logger.log_info("Loading SPN leaf node settings...")

    spn_settings_joint = generateSPNSettings(config_dataset, device)
    spn_settings_marginal = torch.clone(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    logger.log_info("Setting SPN leaf nodes...")

    spn_joint.set_leaf_nodes(spn_settings_joint)
    spn_marginal.set_leaf_nodes(spn_settings_marginal)

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depths: " + str(spn_joint.depth) + ".")
        logger.log_info("SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    test(model_decomposed, spn_joint, spn_marginal, dataset_test, config_dataset, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
