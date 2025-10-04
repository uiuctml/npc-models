#!/usr/bin/env python3

import argparse
import dataset
import header
import json
import logger
import model
import os
import spn
import test_dl
import test_spn
import torch
import tqdm
import utility
import wandb

def computeMPECorrectness(mpe_attributes, labels_decomposed):
    mpe_correctness = []

    for batch_index in range(len(mpe_attributes)):
        correct = True

        for (attribute_index, category_index) in enumerate(mpe_attributes[batch_index]):
            if labels_decomposed[attribute_index][batch_index][category_index] <= 0:
                correct = False
                break

        mpe_correctness.append(correct)

    return mpe_correctness

def getMatrixAColIndicesAttributeIndicesMaps(matrix_a_cols, spn_settings):
    attribute_indices_list = spn_settings[:matrix_a_cols, :-1].cpu().int().tolist()
    matrix_a_col_indices_to_attribute_indices = {}
    attribute_indices_to_matrix_a_col_indices = {}

    for (matrix_a_row_index, attribute_indices) in enumerate(attribute_indices_list):
        matrix_a_col_indices_to_attribute_indices[matrix_a_row_index] = tuple(attribute_indices)
        attribute_indices_to_matrix_a_col_indices[tuple(attribute_indices)] = matrix_a_row_index

    return (matrix_a_col_indices_to_attribute_indices, attribute_indices_to_matrix_a_col_indices)

def findMPEs(matrix_a, matrix_b, predictions_composed, spn_settings):
    (matrix_a_col_indices_to_attribute_indices, _) = getMatrixAColIndicesAttributeIndicesMaps(matrix_a.shape[1], spn_settings)
    mpe_attributes = []

    matrix_a = matrix_a.t() # product of category size of all attributes x number of original labels
    matrix_a = torch.index_select(matrix_a, 1, predictions_composed)   # product of category size of all attributes x batch size
    matrix_c = matrix_a * matrix_b  # product of category size of all attributes x batch size
    mpes_matrix_a_col_indices = torch.argmax(matrix_c, 0)   # 0 x batch size

    for mpes_matrix_a_col_index in mpes_matrix_a_col_indices.cpu().tolist():
        mpe_attributes.append(matrix_a_col_indices_to_attribute_indices[mpes_matrix_a_col_index])

    return mpe_attributes

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name-attribute", type = str, default = "", help = "Attribute run name.", required = True)
    parser.add_argument("-p", "--run-name-pc", type = str, default = "", help = "PC run name.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Random seed.")
    arguments = parser.parse_args()

    test_dl.initializeRunName(arguments.run_name_attribute)

    if arguments.run_name_pc == "":
        logger.log_info("Proceeding without PC run name.")
        header.run_name_spn == ""
        header.config_spn["file_name_checkpoint"] = ""
        header.config_spn["file_name_checkpoint_best"] = ""
        header.config_spn["run_name"] = ""
    else:
        test_spn.initializeRunName(arguments.run_name_pc, "pgd")

    if arguments.seed is not None:
        header.config_decomposed["seed"] = arguments.seed
        header.config_spn["seed"] = arguments.seed

    logger.log_trace("Attribute run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("PC run name: \"" + header.run_name_spn + "\".")
    logger.log_trace("Random seed: " + str(header.config_decomposed["seed"]) + ".")

    return

def saveMPEs(input_file_paths, mpes, mpe_attributes, mpe_correctness, dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_composed):
    batch_size = labels_original.nelement()
    config_dataset = dataset.config

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        mpes[input_file_path] = {}
        mpes[input_file_path]["mpe"] = {}
        mpes[input_file_path]["ground_truth"] = {}
        mpes[input_file_path]["prediction"] = {}

        for (attribute_index, attribute) in enumerate(config_dataset["attributes"]):
            attribute_name = attribute["name"]
            masks_positive_labels_decomposed = (labels_decomposed[attribute_index] > 0)
            counts_positive_labels_decomposed = torch.sum(masks_positive_labels_decomposed, dim = 1)
            count_positive_labels_decomposed = counts_positive_labels_decomposed[batch_index]

            (outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label) = torch.topk(outputs_decomposed[attribute_index][batch_index], count_positive_labels_decomposed)

            mpes[input_file_path]["mpe"][attribute_name] = dataset.classes[attribute_name][mpe_attributes[batch_index][attribute_index]]
            mpes[input_file_path]["ground_truth"][attribute_name] = []
            mpes[input_file_path]["prediction"][attribute_name] = {}

            for (label_decomposed_index, label_decomposed) in enumerate(labels_decomposed[attribute_index][batch_index]):
                if label_decomposed > 0:
                    mpes[input_file_path]["ground_truth"][attribute_name].append(dataset.classes[attribute_name][label_decomposed_index])

            for (output_decomposed_mpe_probability, output_decomposed_mpe_label) in zip(outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label):
                mpes[input_file_path]["prediction"][attribute_name][dataset.classes[attribute_name][output_decomposed_mpe_label.item()]] = output_decomposed_mpe_probability.item()

            mpes[input_file_path]["ground_truth"][attribute_name] = sorted(mpes[input_file_path]["ground_truth"][attribute_name])
            mpes[input_file_path]["prediction"][attribute_name] = dict(sorted(mpes[input_file_path]["prediction"][attribute_name].items()))

        (outputs_composed_mpe_probability, outputs_composed_mpe_label) = torch.max(outputs_composed[batch_index], 0)

        mpes[input_file_path]["mpe"]["correct"] = mpe_correctness[batch_index]
        mpes[input_file_path]["ground_truth"]["original"] = dataset.classes_original[labels_original[batch_index]]
        mpes[input_file_path]["prediction"]["original"] = (dataset.classes_original[outputs_composed_mpe_label], outputs_composed_mpe_probability.item())

    return

def test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["file_name_checkpoint_best"], model_decomposed)

    if header.config_spn["run_name"] != "":
        utility.loadCheckpointBestSPN(header.config_spn["file_name_checkpoint_best"], spn_joint)
        utility.loadCheckpointBestSPN(header.config_spn["file_name_checkpoint_best"], spn_marginal)

    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    mpe_correctness_epoch = []
    mpe_correctness_prediction_correct_epoch = []
    mpe_correctness_prediction_incorrect_epoch = []
    mpes = {}
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    spn_output_rows = len(data_loader.dataset.classes_original)
    spn_output_cols = 1
    tv_distance_epoch = 0
    tv_distances_epoch = []

    for attribute in data_loader.dataset.config["attributes"]:
        attribute_labels = attribute["labels"]
        spn_output_cols *= len(attribute_labels)
        tv_distances_epoch.append(0)

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_decomposed, labels_original, input_file_paths)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            for i in range(len(labels_decomposed)):
                labels_decomposed[i] = labels_decomposed[i].to(device, non_blocking = True)

            (output_decomposed, _) = model_decomposed(input)

            output_decomposed = utility.applySoftmaxDecomposed(output_decomposed)
            (matrix_a, matrix_b, output_composed) = utility.compose(output_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)
            (_, predictions_composed) = torch.max(output_composed, 1)

            prediction_correctness = (predictions_composed == labels_original)
            corrects_composed = torch.sum(prediction_correctness).item()

            accuracy_attribute_batch = utility.computeAccuracyDecomposed(output_decomposed, labels_decomposed, device)
            accuracy_task_batch = corrects_composed / input.size(0)

            accuracy_attribute_epoch += accuracy_attribute_batch
            accuracy_task_epoch += corrects_composed

            for i in range(len(data_loader.dataset.config["attributes"])):
                tv_distance_batch = 0.5 * torch.sum(torch.abs(output_decomposed[i] - labels_decomposed[i]), dim = 1)
                tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

            if header.composed_mpe_find:
                mpe_attributes = findMPEs(matrix_a, matrix_b, predictions_composed, spn_settings_joint)
                mpe_correctness = computeMPECorrectness(mpe_attributes, labels_decomposed)

                if header.composed_mpe_save:
                    saveMPEs(input_file_paths, mpes, mpe_attributes, mpe_correctness, data_loader.dataset, labels_decomposed, labels_original, output_decomposed, output_composed)

                mpe_correctness = torch.tensor(mpe_correctness).to(device)
                mpe_correctness_epoch += mpe_correctness.tolist()
                mpe_correctness_prediction_correct_epoch += mpe_correctness[prediction_correctness].tolist()
                mpe_correctness_prediction_incorrect_epoch += mpe_correctness[~prediction_correctness].tolist()

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy_attribute": accuracy_attribute_batch})
            wandb.log({"testing/batch/accuracy_task": accuracy_task_batch})
            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    if header.composed_mpe_find:
        if not os.path.isdir(header.composed_mpe_dir_output):
            os.makedirs(header.composed_mpe_dir_output, exist_ok = True)

        with open(os.path.join(header.composed_mpe_dir_output, header.composed_mpe_file_name), "w") as file_mpe:
            json.dump(mpes, file_mpe, indent = 4)

    accuracy_attribute_epoch /= len(data_loader)
    accuracy_task_epoch /= len(data_loader.dataset)

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distances_epoch[i] /= len(data_loader.dataset)

    tv_distance_epoch = sum(tv_distances_epoch) / len(tv_distances_epoch)

    wandb.log({"testing/epoch/accuracy_attribute": accuracy_attribute_epoch})
    wandb.log({"testing/epoch/accuracy_task": accuracy_task_epoch})
    wandb.log({"testing/epoch/tv_distance_attribute": tv_distance_epoch})

    wandb.summary["testing/epoch/accuracy_attribute"] = accuracy_attribute_epoch
    wandb.summary["testing/epoch/accuracy_task"] = accuracy_task_epoch
    wandb.summary["testing/epoch/tv_distance_attribute"] = tv_distance_epoch

    logger.log_info("Testing attribute TV distance: " + str(tv_distance_epoch) + ".")
    logger.log_info("Testing attribute accuracy: " + str(accuracy_attribute_epoch) + ".")
    logger.log_info("Testing task accuracy: " + str(accuracy_task_epoch) + ".")

    if header.composed_mpe_find:
        if len(mpe_correctness_epoch) == 0:
            mpe_correctness_epoch = "N/A"
        else:
            mpe_correctness_epoch = sum(mpe_correctness_epoch) / len(mpe_correctness_epoch)

        if len(mpe_correctness_prediction_correct_epoch) == 0:
            mpe_correctness_prediction_correct_epoch = "N/A"
        else:
            mpe_correctness_prediction_correct_epoch = sum(mpe_correctness_prediction_correct_epoch) / len(mpe_correctness_prediction_correct_epoch)

        if len(mpe_correctness_prediction_incorrect_epoch) == 0:
            mpe_correctness_prediction_incorrect_epoch = "N/A"
        else:
            mpe_correctness_prediction_incorrect_epoch = sum(mpe_correctness_prediction_incorrect_epoch) / len(mpe_correctness_prediction_incorrect_epoch)

        logger.log_info("MPE correctness on all predictions: " + str(mpe_correctness_epoch) + ".")
        logger.log_info("MPE correctness on correct predictions: " + str(mpe_correctness_prediction_correct_epoch) + ".")
        logger.log_info("MPE correctness on incorrect predictions: " + str(mpe_correctness_prediction_incorrect_epoch) + ".")

    return

def main():
    processArguments()

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    config = {
        "decomposed": header.config_decomposed,
        "spn": header.config_spn
    }

    wandb.init(config = config, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.NPCDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    device_spn = torch.device("cuda")

    if header.composed_spn_on_cpu:
        logger.log_info("Computing SPNs on CPU.")
        device_spn = torch.device("cpu")

    model_decomposed = model.ResNet34MTL(dataset_test.config, device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    spn_joint = spn.SPN(device_spn)
    spn_marginal = spn.SPN(device_spn)

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])

    logger.log_info("Loading SPN leaf node settings...")

    spn_settings_joint = utility.generateSPNSettings(config_dataset, device)
    spn_settings_marginal = torch.clone(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    logger.log_info("Setting SPN leaf nodes...")

    spn_joint.set_leaf_nodes_categorical(spn_settings_joint)
    spn_marginal.set_leaf_nodes_categorical(spn_settings_marginal)

    logger.log_trace("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
    logger.log_trace("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
    logger.log_trace("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
    logger.log_trace("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
    logger.log_trace("SPN depth: " + str(spn_joint.depth) + ".")
    logger.log_trace("SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
