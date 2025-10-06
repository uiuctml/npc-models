#!/usr/bin/env python3

import argparse
import dataset
import header
import json
import logger
import model
import os
import pc
import test_neural
import test_pc
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

def computeNPCOutput(outputs_decomposed, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device):
    log_likelihoods_joint = pc_joint.forward().to(device)
    log_likelihoods_marginal = pc_marginal.forward().to(device)

    # Compute matrix A and set entries with zero joint and marginal probabilities to zero
    mask_joint = (log_likelihoods_joint == -float("inf"))
    mask_marginal = (log_likelihoods_joint == -float("inf"))
    mask_matrix_a = mask_joint & mask_marginal
    matrix_a = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)
    matrix_a[mask_matrix_a] = 0
    matrix_a = matrix_a.reshape(pc_output_rows, pc_output_cols)

    batch_size = outputs_decomposed[0].shape[0]
    matrix_b_list = []

    for batch in range(batch_size):
        matrix_b_batch = outputs_decomposed[0][batch]

        for task_index in range(1, len(outputs_decomposed)):
            matrix_b_batch = torch.outer(matrix_b_batch, outputs_decomposed[task_index][batch]).flatten()

        matrix_b_list.append(matrix_b_batch)

    matrix_b = torch.stack(matrix_b_list, dim = 0).t()
    matrix_b = matrix_b.to(device)

    matrix_c = torch.matmul(matrix_a, matrix_b).t()

    return (matrix_a, matrix_b, matrix_c)

def findCE(outputs_decomposed_original, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, labels_original, device):
    with torch.set_grad_enabled(True):
        batch_size = labels_original.nelement()
        outputs_decomposed = []
        outputs_decomposed_original = utility.applySoftmaxDecomposed(outputs_decomposed_original)
        progress_bar = tqdm.tqdm(total = batch_size, position = 1, leave = False)
        progress_bar.set_description_str("[INFO]: Optimizing CE attributes")

        for i in range(len(outputs_decomposed_original)):
            outputs_decomposed.append(outputs_decomposed_original[i].detach().clone().requires_grad_(True))

        for _ in range(header.npc_interpret_ce_steps):
            (_, _, outputs_composed_original) = computeNPCOutput(outputs_decomposed, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)
            outputs_composed_prediction = torch.max(outputs_composed_original, 1)[1]
            outputs_composed_indices = torch.where(outputs_composed_prediction != labels_original)[0]

            progress_bar.n = batch_size - outputs_composed_indices.nelement()
            progress_bar.refresh()

            if outputs_composed_indices.nelement() == 0:
                break

            outputs_composed = outputs_composed_original.t()   # number of original labels x batch size
            outputs_composed = outputs_composed[labels_original, torch.arange(outputs_composed.shape[1])]  # 1 x batch size
            outputs_composed = torch.log(outputs_composed)   # 1 x batch size
            outputs_composed = torch.sum(outputs_composed[outputs_composed_indices], 0) # 1 x 1
            outputs_composed.backward(retain_graph = True)

            with torch.set_grad_enabled(False):
                for i in range(len(outputs_decomposed)):
                    outputs_decomposed[i] += header.npc_interpret_ce_learning_rate * outputs_decomposed[i].grad

            rhos = []

            for i in range(len(outputs_decomposed)):
                rho_rhs = outputs_decomposed[i].detach().clone().requires_grad_(False)
                rho_rhs = torch.sort(rho_rhs, descending = True, dim = 1)[0]
                rho_lhs = rho_rhs.clone()   # batch size x attribute category size

                rho_rhs = torch.cumsum(rho_rhs, dim = 1)
                rho_rhs -= 1
                rho_rhs /= torch.arange(1, rho_rhs.shape[1] + 1).to(device) # batch size x attribute category size

                rho = torch.sum(rho_lhs > rho_rhs, dim = 1, keepdim = True)   # batch size x 1
                rhos.append(rho)

            lambdas = []

            for i in range(len(rhos)):
                output_decomposed = outputs_decomposed[i].detach().clone().requires_grad_(False)
                lambda_i = torch.sort(output_decomposed, descending = True, dim = 1)[0] # batch size x attribute category size

                lambda_i_row_indices = torch.arange(lambda_i.shape[1]).long().expand_as(lambda_i).to(device)
                rho = rhos[i].expand_as(lambda_i)
                lambda_i_zero_mask = lambda_i_row_indices >= rho
                lambda_i[lambda_i_zero_mask] = 0

                lambda_i = torch.sum(lambda_i, dim = 1, keepdim = True) # batch size x 1
                lambda_i = 1 - lambda_i
                lambda_i /= rhos[i]

                lambdas.append(lambda_i)

            with torch.set_grad_enabled(False):
                for i in range(len(outputs_decomposed)):
                    outputs_decomposed[i] = torch.clamp(outputs_decomposed[i] + lambdas[i], min = 0)

                    if torch.sum(torch.isnan(outputs_decomposed[i])) > 0:
                        rows_nan = torch.isnan(outputs_decomposed[i]).any(dim = 1)
                        outputs_decomposed[i][rows_nan] = outputs_decomposed_original[i][rows_nan]

                    if torch.sum(torch.isinf(outputs_decomposed[i])) > 0:
                        rows_nan = torch.isinf(outputs_decomposed[i]).any(dim = 1)
                        outputs_decomposed[i][rows_nan] = outputs_decomposed_original[i][rows_nan]

            for i in range(len(outputs_decomposed)):
                outputs_decomposed[i] = outputs_decomposed[i].detach().clone().requires_grad_(True)

        progress_bar.close()

        return outputs_decomposed

def findMPE(matrix_a, matrix_b, predictions_composed, pc_settings):
    attribute_indices_list = pc_settings[:matrix_a.shape[1], :-1].cpu().int().tolist()
    matrix_a_col_indices_to_attribute_indices = {}
    mpe_attributes = []

    for (matrix_a_row_index, attribute_indices) in enumerate(attribute_indices_list):
        matrix_a_col_indices_to_attribute_indices[matrix_a_row_index] = tuple(attribute_indices)

    matrix_a = matrix_a.t() # product of category size of all attributes x number of original labels
    matrix_a = torch.index_select(matrix_a, 1, predictions_composed)   # product of category size of all attributes x batch size
    matrix_c = matrix_a * matrix_b  # product of category size of all attributes x batch size
    mpe_matrix_a_col_indices = torch.argmax(matrix_c, 0)   # 0 x batch size

    for mpe_matrix_a_col_index in mpe_matrix_a_col_indices.cpu().tolist():
        mpe_attributes.append(matrix_a_col_indices_to_attribute_indices[mpe_matrix_a_col_index])

    return mpe_attributes

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name-attribute", type = str, default = "", help = "Attribute run name.", required = True)
    parser.add_argument("-p", "--run-name-pc", type = str, default = "", help = "PC run name.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Random seed.")
    arguments = parser.parse_args()

    test_neural.initializeRunName(arguments.run_name_attribute)

    if arguments.run_name_pc == "":
        logger.log_info("Proceeding without PC run name.")
        header.config_pc["file_name_checkpoint"] = ""
        header.config_pc["file_name_checkpoint_best"] = ""
        header.config_pc["run_name"] = ""
    else:
        test_pc.initializeRunName(arguments.run_name_pc, "pgd")

    if arguments.seed is not None:
        header.config_neural["seed"] = arguments.seed
        header.config_pc["seed"] = arguments.seed

    logger.log_trace("Attribute run name: \"" + header.config_neural["run_name"] + "\".")
    logger.log_trace("PC run name: \"" + header.config_pc["run_name"] + "\".")
    logger.log_trace("Random seed: " + str(header.config_neural["seed"]) + ".")

    return

def recordCE(input_file_paths, interpret, dataset, labels_decomposed, labels_original, outputs_decomposed_ce, outputs_composed_ce):
    batch_size = labels_original.nelement()

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        interpret[input_file_path]["ce"] = {}

        for (attribute_index, attribute) in enumerate(dataset.config["attributes"]):
            attribute_name = attribute["name"]
            masks_positive_labels_decomposed = (labels_decomposed[attribute_index] > 0)
            counts_positive_labels_decomposed = torch.sum(masks_positive_labels_decomposed, dim = 1)
            count_positive_labels_decomposed = counts_positive_labels_decomposed[batch_index]

            (outputs_decomposed_ce_prediction_probability, outputs_decomposed_ce_prediction_label) = torch.topk(outputs_decomposed_ce[attribute_index][batch_index], count_positive_labels_decomposed)

            interpret[input_file_path]["ce"][attribute_name] = {}

            for (output_decomposed_ce_prediction_probability, output_decomposed_ce_prediction_label) in zip(outputs_decomposed_ce_prediction_probability, outputs_decomposed_ce_prediction_label):
                interpret[input_file_path]["ce"][attribute_name][dataset.classes[attribute_name][output_decomposed_ce_prediction_label.item()]] = output_decomposed_ce_prediction_probability.item()

            interpret[input_file_path]["ce"][attribute_name] = dict(sorted(interpret[input_file_path]["ce"][attribute_name].items()))

        (outputs_composed_ce_prediction_probability, outputs_composed_ce_prediction_label) = torch.max(outputs_composed_ce[batch_index], 0)

        interpret[input_file_path]["ce"]["original"] = {dataset.classes_original[outputs_composed_ce_prediction_label]: outputs_composed_ce_prediction_probability.item()}

    return

def recordMPE(input_file_paths, interpret, mpe_attributes, mpe_correctness, dataset, labels_original):
    batch_size = labels_original.nelement()

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        interpret[input_file_path]["mpe"] = {}

        for (attribute_index, attribute) in enumerate(dataset.config["attributes"]):
            attribute_name = attribute["name"]
            interpret[input_file_path]["mpe"][attribute_name] = dataset.classes[attribute_name][mpe_attributes[batch_index][attribute_index]]

        interpret[input_file_path]["mpe"]["correct"] = mpe_correctness[batch_index]

    return

def recordPredictions(input_file_paths, interpret, dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_composed):
    batch_size = labels_original.nelement()

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        interpret[input_file_path] = {}
        interpret[input_file_path]["ground_truth"] = {}
        interpret[input_file_path]["prediction"] = {}

        for (attribute_index, attribute) in enumerate(dataset.config["attributes"]):
            attribute_name = attribute["name"]
            masks_positive_labels_decomposed = (labels_decomposed[attribute_index] > 0)
            counts_positive_labels_decomposed = torch.sum(masks_positive_labels_decomposed, dim = 1)
            count_positive_labels_decomposed = counts_positive_labels_decomposed[batch_index]

            (outputs_decomposed_prediction_probability, outputs_decomposed_prediction_label) = torch.topk(outputs_decomposed[attribute_index][batch_index], count_positive_labels_decomposed)

            interpret[input_file_path]["ground_truth"][attribute_name] = []
            interpret[input_file_path]["prediction"][attribute_name] = {}

            for (label_decomposed_index, label_decomposed) in enumerate(labels_decomposed[attribute_index][batch_index]):
                if label_decomposed > 0:
                    interpret[input_file_path]["ground_truth"][attribute_name].append(dataset.classes[attribute_name][label_decomposed_index])

            for (output_decomposed_prediction_probability, output_decomposed_prediction_label) in zip(outputs_decomposed_prediction_probability, outputs_decomposed_prediction_label):
                interpret[input_file_path]["prediction"][attribute_name][dataset.classes[attribute_name][output_decomposed_prediction_label.item()]] = output_decomposed_prediction_probability.item()

            interpret[input_file_path]["ground_truth"][attribute_name] = sorted(interpret[input_file_path]["ground_truth"][attribute_name])
            interpret[input_file_path]["prediction"][attribute_name] = dict(sorted(interpret[input_file_path]["prediction"][attribute_name].items()))

        (outputs_composed_prediction_probability, outputs_composed_prediction_label) = torch.max(outputs_composed[batch_index], 0)

        interpret[input_file_path]["ground_truth"]["original"] = dataset.classes_original[labels_original[batch_index]]
        interpret[input_file_path]["prediction"]["original"] = {dataset.classes_original[outputs_composed_prediction_label]: outputs_composed_prediction_probability.item()}

    return

def test(model_decomposed, pc_joint, pc_marginal, pc_settings_joint, data_loader, device, batch_step):
    utility.loadCheckpoint(header.config_neural["file_name_checkpoint_best"], model_decomposed)

    if header.config_pc["run_name"] != "":
        utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_joint, True)
        utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_marginal, True)

    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    ce_correctness_attribute_epoch = 0
    ce_correctness_task_epoch = 0
    ce_instances_corrected = 0
    ce_instances_incorrect = 0
    ce_tv_distance_epoch = 0
    ce_tv_distances_epoch = []
    interpret = {}
    mpe_correctness_epoch = []
    mpe_correctness_prediction_correct_epoch = []
    mpe_correctness_prediction_incorrect_epoch = []
    pc_output_rows = len(data_loader.dataset.classes_original)
    pc_output_cols = 1
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    tv_distance_epoch = 0
    tv_distances_epoch = []

    for attribute in data_loader.dataset.config["attributes"]:
        pc_output_cols *= len(attribute["labels"])
        ce_tv_distances_epoch.append(0)
        tv_distances_epoch.append(0)

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    for (batch_index, (input, labels_decomposed, labels_original, input_file_paths)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels_original = labels_original.to(device, non_blocking = True)

        for i in range(len(labels_decomposed)):
            labels_decomposed[i] = labels_decomposed[i].to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            (outputs_decomposed_original, _) = model_decomposed(input)
            outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed_original)

        (matrix_a, matrix_b, outputs_composed) = computeNPCOutput(outputs_decomposed, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)

        (_, predictions_composed) = torch.max(outputs_composed, 1)
        prediction_correctness = (predictions_composed == labels_original)
        corrects_composed = torch.sum(prediction_correctness).item()

        accuracy_attribute_batch = utility.computeAccuracyDecomposed(outputs_decomposed_original, labels_decomposed, device)
        accuracy_task_batch = corrects_composed / input.size(0)

        accuracy_attribute_epoch += accuracy_attribute_batch
        accuracy_task_epoch += corrects_composed

        for i in range(len(data_loader.dataset.config["attributes"])):
            tv_distance_batch = 0.5 * torch.sum(torch.abs(outputs_decomposed[i] - labels_decomposed[i]), dim = 1)
            tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

        if header.npc_interpret:
            recordPredictions(input_file_paths, interpret, data_loader.dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_composed)

            outputs_decomposed_ce = findCE(outputs_decomposed_original, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, labels_original, device)

            (_, _, outputs_composed_ce) = computeNPCOutput(outputs_decomposed_ce, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)

            recordCE(input_file_paths, interpret, data_loader.dataset, labels_decomposed, labels_original, outputs_decomposed_ce, outputs_composed_ce)

            (_, predictions_composed_ce) = torch.max(outputs_composed_ce, 1)
            prediction_correctness_ce = (predictions_composed_ce == labels_original)
            corrects_composed_ce = torch.sum(prediction_correctness_ce).item()
            incorrects_composed = torch.sum(predictions_composed != labels_original).item()

            ce_correctness_attribute_epoch += utility.computeAccuracyDecomposed(outputs_decomposed_ce, labels_decomposed, device)

            ce_instances_corrected += corrects_composed_ce - corrects_composed
            ce_instances_incorrect += incorrects_composed
            ce_correctness_task_epoch += corrects_composed_ce

            for i in range(len(data_loader.dataset.config["attributes"])):
                ce_tv_distance_batch = 0.5 * torch.sum(torch.abs(outputs_decomposed_ce[i] - outputs_decomposed[i]), dim = 1)
                ce_tv_distances_epoch[i] += torch.sum(ce_tv_distance_batch).item()

            mpe_attributes = findMPE(matrix_a, matrix_b, predictions_composed, pc_settings_joint)
            mpe_correctness = computeMPECorrectness(mpe_attributes, labels_decomposed)

            recordMPE(input_file_paths, interpret, mpe_attributes, mpe_correctness, data_loader.dataset, labels_original)

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

    if header.npc_interpret:
        ce_correctness_attribute_epoch /= len(data_loader)
        ce_correctness_task_epoch /= len(data_loader.dataset)

        for i in range(len(data_loader.dataset.config["attributes"])):
            ce_tv_distances_epoch[i] /= len(data_loader.dataset)

        ce_tv_distance_epoch = sum(ce_tv_distances_epoch) / len(ce_tv_distances_epoch)

        if ce_instances_incorrect != 0:
            logger.log_info("CE correction rate: " + str(ce_instances_corrected / ce_instances_incorrect) + ".")
        else:
            logger.log_info("CE correction rate: N/A.")

        logger.log_info("CE attribute TV distance: " + str(ce_tv_distance_epoch) + ".")
        logger.log_info("CE attribute correctness: " + str(ce_correctness_attribute_epoch) + ".")
        logger.log_info("CE task correctness: " + str(ce_correctness_task_epoch) + ".")

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

        if not os.path.isdir(header.project_dir_outputs_interpret):
            os.makedirs(header.project_dir_outputs_interpret, exist_ok = True)

        with open(os.path.join(header.project_dir_outputs_interpret, header.dataset_prefix + ".json"), "w") as file_interpret:
            json.dump(interpret, file_interpret, indent = 4)
            logger.log_info("Saved interpretions to \"" + os.path.join(header.project_dir_outputs_interpret, header.dataset_prefix + ".json") + "\".")

    return

def main():
    processArguments()

    utility.setSeed(header.config_neural["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    config = {
        "decomposed": header.config_neural,
        "pc": header.config_pc
    }

    wandb.init(config = config, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_neural)
    dataset_test = dataset.NPCDataset(header.config_neural["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_neural["batch_size"], shuffle = False, num_workers = header.config_neural["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    device_pc = torch.device("cuda")

    if header.npc_pc_cpu:
        logger.log_info("Computing PCs on CPU.")
        device_pc = torch.device("cpu")

    model_decomposed = model.ResNet34MTL(dataset_test.config, device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    pc_joint = pc.ProbabilisticCircuit(device_pc)
    pc_marginal = pc.ProbabilisticCircuit(device_pc)

    logger.log_info("Loading PC from \"" + header.config_pc["file_path_pc"] + "\"...")

    pc_joint.load(header.config_pc["file_path_pc"])
    pc_marginal.load(header.config_pc["file_path_pc"])

    logger.log_info("Loading PC leaf node settings...")

    pc_settings_joint = utility.generatePCSettings(dataset_test.config, device)
    pc_settings_marginal = torch.clone(pc_settings_joint)
    pc_settings_marginal[:, -1] = -1

    logger.log_info("Setting PC leaf nodes...")

    pc_joint.set_leaf_nodes_categorical(pc_settings_joint)
    pc_marginal.set_leaf_nodes_categorical(pc_settings_marginal)

    logger.log_trace("Total PC nodes: " + str(len(pc_joint.nodes)) + ".")
    logger.log_trace("Total PC sum nodes: " + str(len(pc_joint.sum_nodes)) + ".")
    logger.log_trace("Total PC product nodes: " + str(len(pc_joint.product_nodes)) + ".")
    logger.log_trace("Total PC leaf nodes: " + str(len(pc_joint.leaf_nodes)) + ".")
    logger.log_trace("PC depth: " + str(pc_joint.depth) + ".")
    logger.log_trace("PC leaf node setting dimension: (" + str(int(pc_settings_joint.shape[0])) + ", " + str(int(pc_settings_joint.shape[1])) + ").")

    test(model_decomposed, pc_joint, pc_marginal, pc_settings_joint, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
