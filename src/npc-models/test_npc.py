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

def findCounterfactual(outputs_decomposed_original, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, labels_original, device):
    batch_size = labels_original.nelement()
    outputs_decomposed = []
    outputs_decomposed_original = utility.applySoftmaxDecomposed(outputs_decomposed_original)
    progress_bar = tqdm.tqdm(total = batch_size, position = 1, leave = False)
    progress_bar.set_description_str("[INFO]: Optimizing attributes")

    for i in range(len(outputs_decomposed_original)):
        outputs_decomposed.append(outputs_decomposed_original[i].detach().clone().requires_grad_(True))

    for _ in range(header.npc_ce_steps):
        (_, _, outputs_composed_original) = utility.compose(outputs_decomposed, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)
        outputs_composed_mpe = torch.max(outputs_composed_original, 1)[1]
        outputs_composed_indices = torch.where(outputs_composed_mpe != labels_original)[0]

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
                outputs_decomposed[i] += header.npc_ce_learning_rate * outputs_decomposed[i].grad

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

def saveCounterfactual(input_file_paths, counterfactual, dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_decomposed_counterfactual, outputs_composed, outputs_composed_counterfactual):
    batch_size = labels_original.nelement()

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        counterfactual[input_file_path] = {}
        counterfactual[input_file_path]["counterfactual"] = {}
        counterfactual[input_file_path]["ground_truth"] = {}
        counterfactual[input_file_path]["prediction"] = {}

        for (attribute_index, attribute) in enumerate(dataset.config["attributes"]):
            attribute_name = attribute["name"]
            masks_positive_labels_decomposed = (labels_decomposed[attribute_index] > 0)
            counts_positive_labels_decomposed = torch.sum(masks_positive_labels_decomposed, dim = 1)
            count_positive_labels_decomposed = counts_positive_labels_decomposed[batch_index]

            (outputs_decomposed_counterfactual_mpe_probability, outputs_decomposed_counterfactual_mpe_label) = torch.topk(outputs_decomposed_counterfactual[attribute_index][batch_index], count_positive_labels_decomposed)
            (outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label) = torch.topk(outputs_decomposed[attribute_index][batch_index], count_positive_labels_decomposed)

            counterfactual[input_file_path]["counterfactual"][attribute_name] = {}
            counterfactual[input_file_path]["ground_truth"][attribute_name] = []
            counterfactual[input_file_path]["prediction"][attribute_name] = {}

            for (output_decomposed_counterfactual_mpe_probability, output_decomposed_counterfactual_mpe_label) in zip(outputs_decomposed_counterfactual_mpe_probability, outputs_decomposed_counterfactual_mpe_label):
                counterfactual[input_file_path]["counterfactual"][attribute_name][dataset.classes[attribute_name][output_decomposed_counterfactual_mpe_label.item()]] = output_decomposed_counterfactual_mpe_probability.item()

            for (label_decomposed_index, label_decomposed) in enumerate(labels_decomposed[attribute_index][batch_index]):
                if label_decomposed > 0:
                    counterfactual[input_file_path]["ground_truth"][attribute_name].append(dataset.classes[attribute_name][label_decomposed_index])

            for (output_decomposed_mpe_probability, output_decomposed_mpe_label) in zip(outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label):
                counterfactual[input_file_path]["prediction"][attribute_name][dataset.classes[attribute_name][output_decomposed_mpe_label.item()]] = output_decomposed_mpe_probability.item()

            counterfactual[input_file_path]["counterfactual"][attribute_name] = dict(sorted(counterfactual[input_file_path]["counterfactual"][attribute_name].items()))
            counterfactual[input_file_path]["ground_truth"][attribute_name] = sorted(counterfactual[input_file_path]["ground_truth"][attribute_name])
            counterfactual[input_file_path]["prediction"][attribute_name] = dict(sorted(counterfactual[input_file_path]["prediction"][attribute_name].items()))

        (outputs_composed_counterfactual_mpe_probability, outputs_composed_counterfactual_mpe_label) = torch.max(outputs_composed_counterfactual[batch_index], 0)
        (outputs_composed_mpe_probability, outputs_composed_mpe_label) = torch.max(outputs_composed[batch_index], 0)

        counterfactual[input_file_path]["counterfactual"]["original"] = {dataset.classes_original[outputs_composed_counterfactual_mpe_label]: outputs_composed_counterfactual_mpe_probability.item()}
        counterfactual[input_file_path]["ground_truth"]["original"] = dataset.classes_original[labels_original[batch_index]]
        counterfactual[input_file_path]["prediction"]["original"] = {dataset.classes_original[outputs_composed_mpe_label]: outputs_composed_mpe_probability.item()}

    return

def saveMPE(input_file_paths, mpe, mpe_attributes, mpe_correctness, dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_composed):
    batch_size = labels_original.nelement()

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        mpe[input_file_path] = {}
        mpe[input_file_path]["mpe"] = {}
        mpe[input_file_path]["ground_truth"] = {}
        mpe[input_file_path]["prediction"] = {}

        for (attribute_index, attribute) in enumerate(dataset.config["attributes"]):
            attribute_name = attribute["name"]
            masks_positive_labels_decomposed = (labels_decomposed[attribute_index] > 0)
            counts_positive_labels_decomposed = torch.sum(masks_positive_labels_decomposed, dim = 1)
            count_positive_labels_decomposed = counts_positive_labels_decomposed[batch_index]

            (outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label) = torch.topk(outputs_decomposed[attribute_index][batch_index], count_positive_labels_decomposed)

            mpe[input_file_path]["mpe"][attribute_name] = dataset.classes[attribute_name][mpe_attributes[batch_index][attribute_index]]
            mpe[input_file_path]["ground_truth"][attribute_name] = []
            mpe[input_file_path]["prediction"][attribute_name] = {}

            for (label_decomposed_index, label_decomposed) in enumerate(labels_decomposed[attribute_index][batch_index]):
                if label_decomposed > 0:
                    mpe[input_file_path]["ground_truth"][attribute_name].append(dataset.classes[attribute_name][label_decomposed_index])

            for (output_decomposed_mpe_probability, output_decomposed_mpe_label) in zip(outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label):
                mpe[input_file_path]["prediction"][attribute_name][dataset.classes[attribute_name][output_decomposed_mpe_label.item()]] = output_decomposed_mpe_probability.item()

            mpe[input_file_path]["ground_truth"][attribute_name] = sorted(mpe[input_file_path]["ground_truth"][attribute_name])
            mpe[input_file_path]["prediction"][attribute_name] = dict(sorted(mpe[input_file_path]["prediction"][attribute_name].items()))

        (outputs_composed_mpe_probability, outputs_composed_mpe_label) = torch.max(outputs_composed[batch_index], 0)

        mpe[input_file_path]["mpe"]["correct"] = mpe_correctness[batch_index]
        mpe[input_file_path]["ground_truth"]["original"] = dataset.classes_original[labels_original[batch_index]]
        mpe[input_file_path]["prediction"]["original"] = (dataset.classes_original[outputs_composed_mpe_label], outputs_composed_mpe_probability.item())

    return

def test(model_decomposed, pc_joint, pc_marginal, pc_settings_joint, data_loader, device, batch_step):
    utility.loadCheckpoint(header.config_neural["file_name_checkpoint_best"], model_decomposed)

    if header.config_pc["run_name"] != "":
        utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_joint, True)
        utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_marginal, True)

    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    counterfactual = {}
    correctness_attribute_epoch = 0
    correctness_task_epoch = 0
    instance_count_corrected = 0
    instance_count_incorrect = 0
    counterfactual_tv_distance_epoch = 0
    counterfactual_tv_distances_epoch = []
    mpe = {}
    mpe_correctness_epoch = []
    mpe_correctness_prediction_correct_epoch = []
    mpe_correctness_prediction_incorrect_epoch = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    pc_output_rows = len(data_loader.dataset.classes_original)
    pc_output_cols = 1
    tv_distance_epoch = 0
    tv_distances_epoch = []

    for attribute in data_loader.dataset.config["attributes"]:
        pc_output_cols *= len(attribute["labels"])
        counterfactual_tv_distances_epoch.append(0)
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

        with torch.set_grad_enabled(True):
            outputs_decomposed_counterfactual = findCounterfactual(outputs_decomposed_original, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, labels_original, device)

        (matrix_a, matrix_b, outputs_composed) = utility.compose(outputs_decomposed, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)
        (_, _, outputs_composed_counterfactual) = utility.compose(outputs_decomposed_counterfactual, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)

        if header.npc_ce_save:
            saveCounterfactual(input_file_paths, counterfactual, data_loader.dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_decomposed_counterfactual, outputs_composed, outputs_composed_counterfactual)

        (_, predictions_composed) = torch.max(outputs_composed, 1)
        (_, predictions_composed_counterfactual) = torch.max(outputs_composed_counterfactual, 1)

        prediction_correctness = (predictions_composed == labels_original)
        prediction_correctness_counterfactual = (predictions_composed_counterfactual == labels_original)
        corrects_composed = torch.sum(prediction_correctness).item()
        corrects_composed_counterfactual = torch.sum(prediction_correctness_counterfactual).item()
        incorrects_composed = torch.sum(predictions_composed != labels_original).item()

        instance_count_corrected += corrects_composed_counterfactual - corrects_composed
        instance_count_incorrect += incorrects_composed

        accuracy_attribute_batch = utility.computeAccuracyDecomposed(outputs_decomposed_original, labels_decomposed, device)
        accuracy_task_batch = corrects_composed / input.size(0)

        accuracy_attribute_epoch += accuracy_attribute_batch
        accuracy_task_epoch += corrects_composed

        for i in range(len(data_loader.dataset.config["attributes"])):
            tv_distance_batch = 0.5 * torch.sum(torch.abs(outputs_decomposed[i] - labels_decomposed[i]), dim = 1)
            tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

        correctness_attribute_epoch += utility.computeAccuracyDecomposed(outputs_decomposed_counterfactual, labels_decomposed, device)
        correctness_task_epoch += corrects_composed_counterfactual

        for i in range(len(data_loader.dataset.config["attributes"])):
            counterfactual_tv_distance_batch = 0.5 * torch.sum(torch.abs(outputs_decomposed_counterfactual[i] - outputs_decomposed[i]), dim = 1)
            counterfactual_tv_distances_epoch[i] += torch.sum(counterfactual_tv_distance_batch).item()

        if header.npc_mpe_find:
            mpe_attributes = findMPE(matrix_a, matrix_b, predictions_composed, pc_settings_joint)
            mpe_correctness = computeMPECorrectness(mpe_attributes, labels_decomposed)

            if header.npc_mpe_save:
                saveMPE(input_file_paths, mpe, mpe_attributes, mpe_correctness, data_loader.dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_composed)

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

    correctness_attribute_epoch /= len(data_loader)
    correctness_task_epoch /= len(data_loader.dataset)

    for i in range(len(data_loader.dataset.config["attributes"])):
        counterfactual_tv_distances_epoch[i] /= len(data_loader.dataset)

    counterfactual_tv_distance_epoch = sum(counterfactual_tv_distances_epoch) / len(counterfactual_tv_distances_epoch)

    if instance_count_incorrect != 0:
        logger.log_info("Counterfactual correction rate: " + str(instance_count_corrected / instance_count_incorrect) + ".")
    else:
        logger.log_info("Counterfactual correction rate: N/A.")

    logger.log_info("Counterfactual attribute TV distance: " + str(counterfactual_tv_distance_epoch) + ".")
    logger.log_info("Counterfactual attribute correctness: " + str(correctness_attribute_epoch) + ".")
    logger.log_info("Counterfactual task correctness: " + str(correctness_task_epoch) + ".")

    if header.npc_ce_save:
        if not os.path.isdir(header.interpret_dir_outputs):
            os.makedirs(header.interpret_dir_outputs, exist_ok = True)

        with open(os.path.join(header.interpret_dir_outputs, header.npc_ce_file_name), "w") as file_counterfactual:
            json.dump(counterfactual, file_counterfactual, indent = 4)
            logger.log_info("Saved counterfactual to \"" + os.path.join(header.interpret_dir_outputs, header.npc_ce_file_name) + "\".")

    if header.npc_mpe_find:
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

        if header.npc_mpe_save:
            if not os.path.isdir(header.interpret_dir_outputs):
                os.makedirs(header.interpret_dir_outputs, exist_ok = True)

            with open(os.path.join(header.interpret_dir_outputs, header.npc_mpe_file_name), "w") as file_mpe:
                json.dump(mpe, file_mpe, indent = 4)
                logger.log_info("Saved MPE to \"" + os.path.join(header.interpret_dir_outputs, header.npc_mpe_file_name) + "\".")

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

    if header.npc_pc_on_cpu:
        logger.log_info("Computing PCs on CPU.")
        device_pc = torch.device("cpu")

    model_decomposed = model.ResNet34MTL(dataset_test.config, device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    pc_joint = pc.SPN(device_pc)
    pc_marginal = pc.SPN(device_pc)

    logger.log_info("Loading PC from \"" + header.config_pc["file_path_pc"] + "\"...")

    pc_joint.load(header.config_pc["file_path_pc"])
    pc_marginal.load(header.config_pc["file_path_pc"])

    logger.log_info("Loading PC leaf node settings...")

    pc_settings_joint = utility.generateSPNSettings(dataset_test.config, device)
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
