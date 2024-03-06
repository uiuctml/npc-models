#!/usr/bin/env python3

import argument
import composition
import dataset
import header
import logger
import model
import spn
import torch
import tqdm
import utility

def explain(outputs_decomposed_original, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, labels_original, input_file_paths, device):
    # TODO 1. Perform forward pass using outputs_decomposed and composition.Composition.spn
    # TODO 2. Index-select P(Y = y' | X) from output_composed where y' are ground-truth labels of the current batch
    # TODO 3. Sum up P(Y = y' | X) for all y' ground-truth labels in the batch
    # TODO 4. Perform backward pass on the sum, which computes partial derivatives of P(Y = y' | X) with respect to all P(A | X) for all batches
    # TODO 5. Set the gradients of P(Y = y' | X) that are already MPE to 1
    # TODO 6. Multiply the gradients to the current outputs_decomposed
    # TODO 7. Repeat until all P(Y = y' | X) in the current batch are MPE
    # TODO 8. The resulting outputs_decomposed contains the counterfactual explanations
    # TODO 9. Reset gradient and repeat the above for all batches
    # TODO Save original outputs_decomposed
    # TODO Save perturbed outputs_decomposed

    batch_size = labels_original.nelement()
    outputs_decomposed = []
    smoothing_epsilon = torch.finfo(torch.float).eps
    progress_bar = tqdm.tqdm(total = batch_size, position = 1, leave = False)
    progress_bar.set_description_str("[INFO]: Optimizing MPEs")

    for i in range(len(outputs_decomposed_original)):
        outputs_decomposed.append(outputs_decomposed_original[i].detach().clone().requires_grad_(True))

    while True:
        (_, _, output_composed_original) = composition.Composition.spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)
        output_composed_mpe = torch.max(output_composed_original, 1)[1]

        logger.log_debug(output_composed_mpe)
        logger.log_debug(output_composed_mpe.shape)

        output_composed_indices = torch.where(output_composed_mpe != labels_original)[0]

        progress_bar.n = batch_size - output_composed_indices.nelement()
        progress_bar.refresh()

        logger.log_debug(output_composed_indices)

        if output_composed_indices.nelement() == 0:
            break

        output_composed = output_composed_original.t()   # number of original labels x batch size
        output_composed = output_composed[labels_original, torch.arange(output_composed.shape[1])]  # 1 x batch size
        output_composed = torch.sum(output_composed[output_composed_indices], 0) # 1 x 1
        output_composed.backward(retain_graph = True)

        with torch.set_grad_enabled(False):
            for i in range(len(outputs_decomposed)):
                torch.set_printoptions(edgeitems = 10)
                logger.log_debug(outputs_decomposed[i].grad)

                outputs_decomposed_grad_row_indices = torch.where(torch.sum(outputs_decomposed[i].grad, 1) == 0)[0]

                logger.log_debug(outputs_decomposed_grad_row_indices)

                outputs_decomposed[i].grad[outputs_decomposed_grad_row_indices, :] = 1

                logger.log_debug(outputs_decomposed[i].grad)

                outputs_decomposed[i] *= outputs_decomposed[i].grad + smoothing_epsilon

                logger.log_debug(outputs_decomposed[i])
                logger.log_debug(torch.sum(outputs_decomposed[i], 1, keepdim = True))
                logger.log_debug(outputs_decomposed[i].shape)
                logger.log_debug(torch.sum(outputs_decomposed[i], 1, keepdim = True).shape)

                outputs_decomposed[i] /= torch.sum(outputs_decomposed[i], 1, keepdim = True) + smoothing_epsilon

                logger.log_debug(outputs_decomposed[i])

        for i in range(len(outputs_decomposed_original)):
            outputs_decomposed[i] = outputs_decomposed[i].detach().clone().requires_grad_(True)

    progress_bar.close()

    return outputs_decomposed

def test(model_decomposed, spn_joint, spn_marginal, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)
    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])

    # TEST
    accuracy_epoch_composed = 0
    accuracy_epoch_composed_counterfactual = 0
    # TEST

    config_dataset = data_loader.dataset.config
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    spn_output_rows = len(data_loader.dataset.classes_original)
    spn_output_cols = 1

    for attribute in config_dataset["attributes"]:
        spn_output_cols *= len(attribute["labels"])

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    for (batch_index, (input, labels_decomposed, labels_original, input_file_paths)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels_decomposed = labels_decomposed.to(device, non_blocking = True)
        labels_original = labels_original.to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            (outputs_decomposed, _) = model_decomposed(input)
            outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed)

        with torch.set_grad_enabled(True):
            outputs_decomposed_counterfactual = explain(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, labels_original, input_file_paths, device)

        (_, _, output_composed) = composition.Composition.spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)
        (_, _, output_composed_counterfactual) = composition.Composition.spn(outputs_decomposed_counterfactual, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)
        (_, predictions_composed) = torch.max(output_composed, 1)
        (_, predictions_composed_counterfactual) = torch.max(output_composed_counterfactual, 1)

        corrects_composed = torch.sum(predictions_composed == labels_original.data).item()
        corrects_composed_counterfactual = torch.sum(predictions_composed_counterfactual == labels_original.data).item()

        accuracy_epoch_composed += corrects_composed
        accuracy_epoch_composed_counterfactual += corrects_composed_counterfactual

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

    progress_bar.close()

    accuracy_epoch_composed /= len(data_loader.dataset)
    accuracy_epoch_composed_counterfactual /= len(data_loader.dataset)

    logger.log_info("Composed testing accuracy: " + str(accuracy_epoch_composed) + ".")
    logger.log_info("Composed counterfactual testing accuracy: " + str(accuracy_epoch_composed_counterfactual) + ".")

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

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    spn_joint = spn.SPN(device)
    spn_marginal = spn.SPN(device)

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])

    logger.log_info("Loading SPN leaf node settings...")

    spn_settings_joint = utility.generateSPNSettings(config_dataset, device)
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

    test(model_decomposed, spn_joint, spn_marginal, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
