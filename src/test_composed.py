#!/usr/bin/env python3

import argument
import dataset
import header
import json
import logger
import model
import os
import spn
import torch
import tqdm
import utility
import wandb

def test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)

    if header.config_spn["run_name"] != "":
        utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
        utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])

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

            if header.composed_find_mpe:
                mpe_attributes = utility.findMPEs(matrix_a, matrix_b, predictions_composed, spn_settings_joint)
                mpe_correctness = utility.computeMPECorrectness(mpe_attributes, labels_decomposed)

                if header.composed_save_mpe:
                    utility.saveMPEs(input_file_paths, mpes, mpe_attributes, mpe_correctness, data_loader.dataset, labels_decomposed, labels_original, output_decomposed, output_composed)

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

    if header.composed_find_mpe:
        if not os.path.isdir(header.dir_output_mpe):
            os.makedirs(header.dir_output_mpe, exist_ok = True)

        with open(os.path.join(header.dir_output_mpe, header.file_name_mpe), "w") as file_mpe:
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

    if header.composed_find_mpe:
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
    argument.processArgumentsTestComposed()

    header.run_name_baseline = header.config_decomposed["run_name"]
    header.config_baseline["dir_dataset_test"] = header.config_decomposed["dir_dataset_test"]
    header.config_baseline["file_name_checkpoint"] = header.run_name_decomposed + ".tar"
    header.config_baseline["file_name_checkpoint_best"] = header.run_name_decomposed + ".best.tar"
    header.config_baseline["run_name"] = header.run_name_decomposed

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
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    device_spn = torch.device("cuda")

    if header.composed_spn_on_cpu:
        logger.log_info("Computing SPNs on CPU.")
        device_spn = torch.device("cpu")

    model_decomposed = model.createModelDecomposed(device)
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

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depth: " + str(spn_joint.depth) + ".")
        logger.log_info("SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
