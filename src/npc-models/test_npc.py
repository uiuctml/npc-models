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

def computeMPEAlignment(mpe_attributes, labels_attribute):
    mpe_alignment = []

    for batch_index in range(len(mpe_attributes)):
        aligned = True

        for (attribute_index, category_index) in enumerate(mpe_attributes[batch_index]):
            if labels_attribute[attribute_index][batch_index][category_index] <= 0:
                aligned = False
                break

        mpe_alignment.append(aligned)

    return mpe_alignment

def computeNPCOutput(outputs_attribute, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device):
    log_likelihoods_joint = pc_joint.forward().to(device)
    log_likelihoods_marginal = pc_marginal.forward().to(device)

    # Compute PC matrix and set entries with zero joint and marginal probabilities to zero
    mask_joint = (log_likelihoods_joint == -float("inf"))
    mask_marginal = (log_likelihoods_joint == -float("inf"))
    mask_matrix_pc = mask_joint & mask_marginal
    matrix_pc = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)
    matrix_pc[mask_matrix_pc] = 0
    matrix_pc = matrix_pc.reshape(pc_output_rows, pc_output_cols)

    batch_size = outputs_attribute[0].shape[0]
    matrix_neural_list = []

    for batch in range(batch_size):
        matrix_neural_batch = outputs_attribute[0][batch]

        for task_index in range(1, len(outputs_attribute)):
            matrix_neural_batch = torch.outer(matrix_neural_batch, outputs_attribute[task_index][batch]).flatten()

        matrix_neural_list.append(matrix_neural_batch)

    matrix_neural = torch.stack(matrix_neural_list, dim = 0).t()
    matrix_neural = matrix_neural.to(device)

    matrix_npc = torch.matmul(matrix_pc, matrix_neural).t()

    return (matrix_pc, matrix_neural, matrix_npc)

def findCE(outputs_attribute_original, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, labels_class, device):
    with torch.set_grad_enabled(True):
        batch_size = labels_class.nelement()
        outputs_attribute = []
        outputs_attribute_original = utility.applySoftmaxAttribute(outputs_attribute_original)
        progress_bar = tqdm.tqdm(total = batch_size, position = 1, leave = False)
        progress_bar.set_description_str("[INFO]: Optimizing CE attributes")

        for i in range(len(outputs_attribute_original)):
            outputs_attribute.append(outputs_attribute_original[i].detach().clone().requires_grad_(True))

        for _ in range(header.npc_interpret_ce_steps):
            (_, _, output_npc_original) = computeNPCOutput(outputs_attribute, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)
            output_npc_prediction = torch.max(output_npc_original, 1)[1]
            output_npc_indices = torch.where(output_npc_prediction != labels_class)[0]

            progress_bar.n = batch_size - output_npc_indices.nelement()
            progress_bar.refresh()

            if output_npc_indices.nelement() == 0:
                break

            output_npc = output_npc_original.t()   # number of class labels x batch size
            output_npc = output_npc[labels_class, torch.arange(output_npc.shape[1])]  # 1 x batch size
            output_npc = torch.log(output_npc)   # 1 x batch size
            output_npc = torch.sum(output_npc[output_npc_indices], 0) # 1 x 1
            output_npc.backward(retain_graph = True)

            with torch.set_grad_enabled(False):
                for i in range(len(outputs_attribute)):
                    outputs_attribute[i] += header.npc_interpret_ce_learning_rate * outputs_attribute[i].grad

            rhos = []

            for i in range(len(outputs_attribute)):
                rho_rhs = outputs_attribute[i].detach().clone().requires_grad_(False)
                rho_rhs = torch.sort(rho_rhs, descending = True, dim = 1)[0]
                rho_lhs = rho_rhs.clone()   # batch size x attribute category size

                rho_rhs = torch.cumsum(rho_rhs, dim = 1)
                rho_rhs -= 1
                rho_rhs /= torch.arange(1, rho_rhs.shape[1] + 1).to(device) # batch size x attribute category size

                rho = torch.sum(rho_lhs > rho_rhs, dim = 1, keepdim = True)   # batch size x 1
                rhos.append(rho)

            lambdas = []

            for i in range(len(rhos)):
                output_attribute = outputs_attribute[i].detach().clone().requires_grad_(False)
                lambda_i = torch.sort(output_attribute, descending = True, dim = 1)[0] # batch size x attribute category size

                lambda_i_row_indices = torch.arange(lambda_i.shape[1]).long().expand_as(lambda_i).to(device)
                rho = rhos[i].expand_as(lambda_i)
                lambda_i_zero_mask = lambda_i_row_indices >= rho
                lambda_i[lambda_i_zero_mask] = 0

                lambda_i = torch.sum(lambda_i, dim = 1, keepdim = True) # batch size x 1
                lambda_i = 1 - lambda_i
                lambda_i /= rhos[i]

                lambdas.append(lambda_i)

            with torch.set_grad_enabled(False):
                for i in range(len(outputs_attribute)):
                    outputs_attribute[i] = torch.clamp(outputs_attribute[i] + lambdas[i], min = 0)

                    if torch.sum(torch.isnan(outputs_attribute[i])) > 0:
                        rows_nan = torch.isnan(outputs_attribute[i]).any(dim = 1)
                        outputs_attribute[i][rows_nan] = outputs_attribute_original[i][rows_nan]

                    if torch.sum(torch.isinf(outputs_attribute[i])) > 0:
                        rows_nan = torch.isinf(outputs_attribute[i]).any(dim = 1)
                        outputs_attribute[i][rows_nan] = outputs_attribute_original[i][rows_nan]

            for i in range(len(outputs_attribute)):
                outputs_attribute[i] = outputs_attribute[i].detach().clone().requires_grad_(True)

        progress_bar.close()

        return outputs_attribute

def findMPE(matrix_pc, matrix_neural, predictions_npc, pc_settings):
    attribute_indices_list = pc_settings[:matrix_pc.shape[1], :-1].cpu().int().tolist()
    matrix_pc_col_indices_to_attribute_indices = {}
    mpe_attributes = []

    for (matrix_pc_row_index, attribute_indices) in enumerate(attribute_indices_list):
        matrix_pc_col_indices_to_attribute_indices[matrix_pc_row_index] = tuple(attribute_indices)

    matrix_pc = matrix_pc.t() # product of category size of all attributes x number of class labels
    matrix_pc = torch.index_select(matrix_pc, 1, predictions_npc)   # product of category size of all attributes x batch size
    matrix_npc = matrix_pc * matrix_neural  # product of category size of all attributes x batch size
    mpe_matrix_pc_col_indices = torch.argmax(matrix_npc, 0)   # 0 x batch size

    for mpe_matrix_pc_col_index in mpe_matrix_pc_col_indices.cpu().tolist():
        mpe_attributes.append(matrix_pc_col_indices_to_attribute_indices[mpe_matrix_pc_col_index])

    return mpe_attributes

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name-neural", type = str, default = "", help = "Neural run name.", required = True)
    parser.add_argument("-p", "--run-name-pc", type = str, default = "", help = "PC run name.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Seed.")
    arguments = parser.parse_args()

    test_neural.initializeRunName(arguments.run_name_neural)

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

    logger.log_trace("Neural run name: \"" + header.config_neural["run_name"] + "\".")
    logger.log_trace("PC run name: \"" + header.config_pc["run_name"] + "\".")
    logger.log_trace("Seed: " + str(header.config_neural["seed"]) + ".")

    return

def recordCE(input_file_paths, interpret, dataset, labels_attribute, labels_class, outputs_attribute_ce, output_npc_ce):
    batch_size = labels_class.nelement()

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        interpret[input_file_path]["ce"] = {}

        for (attribute_index, attribute) in enumerate(dataset.config["attributes"]):
            attribute_name = attribute["name"]
            masks_positive_labels_attribute = (labels_attribute[attribute_index] > 0)
            counts_positive_labels_attribute = torch.sum(masks_positive_labels_attribute, dim = 1)
            count_positive_labels_attribute = counts_positive_labels_attribute[batch_index]

            (outputs_attribute_ce_prediction_probability, outputs_attribute_ce_prediction_label) = torch.topk(outputs_attribute_ce[attribute_index][batch_index], count_positive_labels_attribute)

            interpret[input_file_path]["ce"][attribute_name] = {}

            for (output_attribute_ce_prediction_probability, output_attribute_ce_prediction_label) in zip(outputs_attribute_ce_prediction_probability, outputs_attribute_ce_prediction_label):
                interpret[input_file_path]["ce"][attribute_name][dataset.labels_attribute[attribute_name][output_attribute_ce_prediction_label.item()]] = output_attribute_ce_prediction_probability.item()

            interpret[input_file_path]["ce"][attribute_name] = dict(sorted(interpret[input_file_path]["ce"][attribute_name].items()))

        (output_npc_ce_prediction_probability, output_npc_ce_prediction_label) = torch.max(output_npc_ce[batch_index], 0)

        interpret[input_file_path]["ce"]["class"] = {dataset.labels_class[output_npc_ce_prediction_label]: output_npc_ce_prediction_probability.item()}

    return

def recordMPE(input_file_paths, interpret, mpe_attributes, mpe_alignment, dataset, labels_class):
    batch_size = labels_class.nelement()

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        interpret[input_file_path]["mpe"] = {}

        for (attribute_index, attribute) in enumerate(dataset.config["attributes"]):
            attribute_name = attribute["name"]
            interpret[input_file_path]["mpe"][attribute_name] = dataset.labels_attribute[attribute_name][mpe_attributes[batch_index][attribute_index]]

        interpret[input_file_path]["mpe"]["aligned"] = mpe_alignment[batch_index]

    return

def recordPredictions(input_file_paths, interpret, dataset, labels_attribute, labels_class, outputs_attribute, output_npc):
    batch_size = labels_class.nelement()

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        interpret[input_file_path] = {}
        interpret[input_file_path]["ground_truth"] = {}
        interpret[input_file_path]["prediction"] = {}

        for (attribute_index, attribute) in enumerate(dataset.config["attributes"]):
            attribute_name = attribute["name"]
            masks_positive_labels_attribute = (labels_attribute[attribute_index] > 0)
            counts_positive_labels_attribute = torch.sum(masks_positive_labels_attribute, dim = 1)
            count_positive_labels_attribute = counts_positive_labels_attribute[batch_index]

            (outputs_attribute_prediction_probability, outputs_attribute_prediction_label) = torch.topk(outputs_attribute[attribute_index][batch_index], count_positive_labels_attribute)

            interpret[input_file_path]["ground_truth"][attribute_name] = []
            interpret[input_file_path]["prediction"][attribute_name] = {}

            for (label_attribute_index, label_attribute) in enumerate(labels_attribute[attribute_index][batch_index]):
                if label_attribute > 0:
                    interpret[input_file_path]["ground_truth"][attribute_name].append(dataset.labels_attribute[attribute_name][label_attribute_index])

            for (output_attribute_prediction_probability, output_attribute_prediction_label) in zip(outputs_attribute_prediction_probability, outputs_attribute_prediction_label):
                interpret[input_file_path]["prediction"][attribute_name][dataset.labels_attribute[attribute_name][output_attribute_prediction_label.item()]] = output_attribute_prediction_probability.item()

            interpret[input_file_path]["ground_truth"][attribute_name] = sorted(interpret[input_file_path]["ground_truth"][attribute_name])
            interpret[input_file_path]["prediction"][attribute_name] = dict(sorted(interpret[input_file_path]["prediction"][attribute_name].items()))

        (output_npc_prediction_probability, output_npc_prediction_label) = torch.max(output_npc[batch_index], 0)

        interpret[input_file_path]["ground_truth"]["class"] = dataset.labels_class[labels_class[batch_index]]
        interpret[input_file_path]["prediction"]["class"] = {dataset.labels_class[output_npc_prediction_label]: output_npc_prediction_probability.item()}

    return

def test(model_neural, pc_joint, pc_marginal, pc_settings_joint, data_loader, device, batch_step):
    utility.loadCheckpoint(header.config_neural["file_name_checkpoint_best"], model_neural)

    if header.config_pc["run_name"] != "":
        utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_joint, True)
        utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_marginal, True)

    accuracy_concept_epoch = 0
    accuracy_classification_epoch = 0
    ce_instances_corrected = 0
    ce_instances_incorrect = 0
    interpret = {}
    mpe_alignment_epoch = []
    pc_output_rows = len(data_loader.dataset.labels_class)
    pc_output_cols = 1
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    tv_distance_epoch = 0
    tv_distances_epoch = []

    for attribute in data_loader.dataset.config["attributes"]:
        pc_output_cols *= len(attribute["labels"])
        tv_distances_epoch.append(0)

    model_neural.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    for (batch_index, (input, labels_attribute, labels_class, input_file_paths)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels_class = labels_class.to(device, non_blocking = True)

        for i in range(len(labels_attribute)):
            labels_attribute[i] = labels_attribute[i].to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            (outputs_attribute_original, _) = model_neural(input)
            outputs_attribute = utility.applySoftmaxAttribute(outputs_attribute_original)

        (matrix_pc, matrix_neural, output_npc) = computeNPCOutput(outputs_attribute, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)

        (_, predictions_npc) = torch.max(output_npc, 1)
        prediction_correctness = (predictions_npc == labels_class)
        corrects_npc = torch.sum(prediction_correctness).item()

        accuracy_concept_batch = utility.computeConceptAccuracy(outputs_attribute_original, labels_attribute, device)
        accuracy_classification_batch = corrects_npc / input.size(0)

        accuracy_concept_epoch += accuracy_concept_batch
        accuracy_classification_epoch += corrects_npc

        for i in range(len(data_loader.dataset.config["attributes"])):
            tv_distance_batch = 0.5 * torch.sum(torch.abs(outputs_attribute[i] - labels_attribute[i]), dim = 1)
            tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

        if header.npc_interpret:
            recordPredictions(input_file_paths, interpret, data_loader.dataset, labels_attribute, labels_class, outputs_attribute, output_npc)

            outputs_attribute_ce = findCE(outputs_attribute_original, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, labels_class, device)

            (_, _, output_npc_ce) = computeNPCOutput(outputs_attribute_ce, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)

            recordCE(input_file_paths, interpret, data_loader.dataset, labels_attribute, labels_class, outputs_attribute_ce, output_npc_ce)

            (_, predictions_npc_ce) = torch.max(output_npc_ce, 1)
            prediction_correctness_ce = (predictions_npc_ce == labels_class)
            corrects_npc_ce = torch.sum(prediction_correctness_ce).item()
            incorrects_npc = torch.sum(predictions_npc != labels_class).item()

            ce_instances_corrected += corrects_npc_ce - corrects_npc
            ce_instances_incorrect += incorrects_npc

            mpe_attributes = findMPE(matrix_pc, matrix_neural, predictions_npc, pc_settings_joint)
            mpe_alignment = computeMPEAlignment(mpe_attributes, labels_attribute)

            recordMPE(input_file_paths, interpret, mpe_attributes, mpe_alignment, data_loader.dataset, labels_class)

            mpe_alignment = torch.tensor(mpe_alignment).to(device)
            mpe_alignment_epoch += mpe_alignment[prediction_correctness].tolist()

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

        wandb.log({"testing/batch/accuracy_concept": accuracy_concept_batch})
        wandb.log({"testing/batch/accuracy_classification": accuracy_classification_batch})
        wandb.log({"testing/batch/step": batch_step})

        batch_step += 1

    progress_bar.close()

    accuracy_concept_epoch /= len(data_loader)
    accuracy_classification_epoch /= len(data_loader.dataset)

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distances_epoch[i] /= len(data_loader.dataset)

    tv_distance_epoch = sum(tv_distances_epoch) / len(tv_distances_epoch)

    wandb.log({"testing/epoch/accuracy_concept": accuracy_concept_epoch})
    wandb.log({"testing/epoch/accuracy_classification": accuracy_classification_epoch})
    wandb.log({"testing/epoch/tv_distance": tv_distance_epoch})

    wandb.summary["testing/epoch/accuracy_concept"] = accuracy_concept_epoch
    wandb.summary["testing/epoch/accuracy_classification"] = accuracy_classification_epoch
    wandb.summary["testing/epoch/tv_distance"] = tv_distance_epoch

    logger.log_info("Testing mean TV distance: " + str(tv_distance_epoch) + ".")
    logger.log_info("Testing mean concept accuracy: " + str(accuracy_concept_epoch) + ".")
    logger.log_info("Testing classification accuracy: " + str(accuracy_classification_epoch) + ".")

    if header.npc_interpret:
        if ce_instances_incorrect != 0:
            logger.log_info("CE correction rate: " + str(ce_instances_corrected / ce_instances_incorrect) + ".")
        else:
            logger.log_info("CE correction rate: N/A.")

        if len(mpe_alignment_epoch) != 0:
            logger.log_info("MPE alignment rate: " + str(sum(mpe_alignment_epoch) / len(mpe_alignment_epoch)) + ".")
        else:
            logger.log_info("MPE alignment rate: N/A.")

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
        "neural": header.config_neural,
        "pc": header.config_pc
    }

    wandb.init(config = config, mode = "disabled")

    dataset_transforms = utility.createTransforms(header.config_neural)
    dataset_test = dataset.NPCDataset(header.config_neural["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_neural["batch_size"], shuffle = False, num_workers = header.config_neural["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    device_pc = torch.device("cuda")

    if header.npc_pc_cpu:
        logger.log_info("Computing PCs on CPU.")
        device_pc = torch.device("cpu")

    model_neural = model.ResNet34MTL(dataset_test.config, device)
    model_neural = torch.nn.DataParallel(model_neural)
    model_neural = model_neural.to(device)
    pc_joint = pc.ProbabilisticCircuit(device_pc)
    pc_marginal = pc.ProbabilisticCircuit(device_pc)

    logger.log_info("Loading PC \"" + header.config_pc["file_path_pc"] + "\"...")

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

    test(model_neural, pc_joint, pc_marginal, pc_settings_joint, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
