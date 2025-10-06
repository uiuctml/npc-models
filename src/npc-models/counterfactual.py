#!/usr/bin/env python3

import dataset
import header
import json
import logger
import model
import os
import pc
import test_npc
import torch
import tqdm
import utility

def computeCounterfactual(outputs_decomposed_original, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, labels_original, device):
    batch_size = labels_original.nelement()
    outputs_decomposed = []
    outputs_decomposed_original = utility.applySoftmaxDecomposed(outputs_decomposed_original)
    progress_bar = tqdm.tqdm(total = batch_size, position = 1, leave = False)
    progress_bar.set_description_str("[INFO]: Optimizing attributes")

    for i in range(len(outputs_decomposed_original)):
        outputs_decomposed.append(outputs_decomposed_original[i].detach().clone().requires_grad_(True))

    for _ in range(header.counterfactual_steps):
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
                outputs_decomposed[i] += header.counterfactual_learning_rate * outputs_decomposed[i].grad

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

def test(model_decomposed, pc_joint, pc_marginal, data_loader, device, batch_step):
    utility.loadCheckpoint(header.config_neural["file_name_checkpoint_best"], model_decomposed)

    if header.config_pc["run_name"] != "":
        utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_joint, True)
        utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_marginal, True)

    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    correctness_attribute_epoch = 0
    correctness_task_epoch = 0
    instance_count_corrected = 0
    instance_count_incorrect = 0
    tv_distance_epoch = 0
    tv_distances_epoch = []
    counterfactual_tv_distance_epoch = 0
    counterfactual_tv_distances_epoch = []

    counterfactual = {}
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    pc_output_rows = len(data_loader.dataset.classes_original)
    pc_output_cols = 1

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
            outputs_decomposed_counterfactual = computeCounterfactual(outputs_decomposed_original, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, labels_original, device)

        (_, _, outputs_composed) = utility.compose(outputs_decomposed, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)
        (_, _, outputs_composed_counterfactual) = utility.compose(outputs_decomposed_counterfactual, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)

        if header.counterfactual_save:
            saveCounterfactual(input_file_paths, counterfactual, data_loader.dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_decomposed_counterfactual, outputs_composed, outputs_composed_counterfactual)

        (_, predictions_composed) = torch.max(outputs_composed, 1)
        (_, predictions_composed_counterfactual) = torch.max(outputs_composed_counterfactual, 1)

        corrects_composed = torch.sum(predictions_composed == labels_original).item()
        corrects_composed_counterfactual = torch.sum(predictions_composed_counterfactual == labels_original).item()
        incorrects_composed = torch.sum(predictions_composed != labels_original).item()

        instance_count_corrected += corrects_composed_counterfactual - corrects_composed
        instance_count_incorrect += incorrects_composed

        accuracy_attribute_epoch += utility.computeAccuracyDecomposed(outputs_decomposed_original, labels_decomposed, device)
        accuracy_task_epoch += corrects_composed

        for i in range(len(data_loader.dataset.config["attributes"])):
            tv_distance_batch = 0.5 * torch.sum(torch.abs(outputs_decomposed[i] - labels_decomposed[i]), dim = 1)
            tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

        correctness_attribute_epoch += utility.computeAccuracyDecomposed(outputs_decomposed_counterfactual, labels_decomposed, device)
        correctness_task_epoch += corrects_composed_counterfactual

        for i in range(len(data_loader.dataset.config["attributes"])):
            counterfactual_tv_distance_batch = 0.5 * torch.sum(torch.abs(outputs_decomposed_counterfactual[i] - outputs_decomposed[i]), dim = 1)
            counterfactual_tv_distances_epoch[i] += torch.sum(counterfactual_tv_distance_batch).item()

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    accuracy_task_epoch /= len(data_loader.dataset)

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distances_epoch[i] /= len(data_loader.dataset)

    tv_distance_epoch = sum(tv_distances_epoch) / len(tv_distances_epoch)

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

    if header.counterfactual_save:
        if not os.path.isdir(header.counterfactual_dir_outputs):
            os.makedirs(header.counterfactual_dir_outputs, exist_ok = True)

        with open(os.path.join(header.counterfactual_dir_outputs, header.counterfactual_file_name), "w") as file_counterfactual:
            json.dump(counterfactual, file_counterfactual, indent = 4)
            logger.log_info("Saved counterfactual to \"" + os.path.join(header.counterfactual_dir_outputs, header.counterfactual_file_name) + "\".")

    return

def main():
    test_npc.processArguments()

    utility.setSeed(header.config_neural["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

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

    test(model_decomposed, pc_joint, pc_marginal, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
