#!/usr/bin/env python3

import argument
import dataset
import header
import json
import logger
import model
import os
import qpsolvers
import spn
import torch
import tqdm
import utility

# 1. Perform forward pass using outputs_decomposed and composition.Composition.spn
# 2. Index-select P(Y = y' | X) from outputs_composed where y' are ground-truth labels of the current batch
# 3. Sum up P(Y = y' | X) for all y' ground-truth labels in the batch
# 4. Perform backward pass on the sum, which computes partial derivatives of P(Y = y' | X) with respect to all P(A | X) for all batches
# 5. If gradients are multiplicative, set the gradients of P(Y = y' | X) that are already MPE to 1
# 6. Apply the gradients to the current outputs_decomposed
# 7. Repeat until all P(Y = y' | X) in the current batch are MPE
# 8. The resulting outputs_decomposed contains the counterfactual explanations
# 9. Reset gradient and repeat the above for all batches

def counterfactual_cccp(outputs_decomposed_original, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, labels_original, device):
    batch_size = labels_original.nelement()
    outputs_decomposed = []
    outputs_decomposed_original = utility.applySoftmaxDecomposed(outputs_decomposed_original)
    smoothing_epsilon = torch.finfo(torch.float).eps
    progress_bar = tqdm.tqdm(total = batch_size, position = 1, leave = False)
    progress_bar.set_description_str("[INFO]: Optimizing attributes")

    for i in range(len(outputs_decomposed_original)):
        outputs_decomposed.append(outputs_decomposed_original[i].detach().clone().requires_grad_(True))

    for _ in range(header.counterfactual_steps):
        (_, _, outputs_composed_original) = utility.compose(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)
        outputs_composed_mpe = torch.max(outputs_composed_original, 1)[1]
        outputs_composed_indices = torch.where(outputs_composed_mpe != labels_original)[0]

        progress_bar.n = batch_size - outputs_composed_indices.nelement()
        progress_bar.refresh()

        if outputs_composed_indices.nelement() == 0:
            break

        outputs_composed = outputs_composed_original.t()   # number of original labels x batch size
        outputs_composed = outputs_composed[labels_original, torch.arange(outputs_composed.shape[1])]  # 1 x batch size
        outputs_composed = torch.sum(outputs_composed[outputs_composed_indices], 0) # 1 x 1
        outputs_composed.backward(retain_graph = True)

        with torch.set_grad_enabled(False):
            for i in range(len(outputs_decomposed)):
                outputs_decomposed_grad_row_indices = torch.where(torch.sum(outputs_decomposed[i].grad, 1) == 0)[0]
                outputs_decomposed[i].grad[outputs_decomposed_grad_row_indices, :] = 1
                outputs_decomposed[i] *= outputs_decomposed[i].grad + smoothing_epsilon
                outputs_decomposed[i] /= torch.sum(outputs_decomposed[i], 1, keepdim = True) + smoothing_epsilon

        for i in range(len(outputs_decomposed)):
            outputs_decomposed[i] = outputs_decomposed[i].detach().clone().requires_grad_(True)

    progress_bar.close()

    return (outputs_decomposed, None, None, None, None)

def counterfactual_gd(outputs_decomposed_original, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, labels_original, device):
    batch_size = labels_original.nelement()
    outputs_decomposed = []
    progress_bar = tqdm.tqdm(total = batch_size, position = 1, leave = False)
    progress_bar.set_description_str("[INFO]: Optimizing attributes")

    for i in range(len(outputs_decomposed_original)):
        outputs_decomposed.append(outputs_decomposed_original[i].detach().clone().requires_grad_(True))

    for _ in range(header.counterfactual_steps):
        outputs_decomposed_softmax = utility.applySoftmaxDecomposed(outputs_decomposed)
        (_, _, outputs_composed_original) = utility.compose(outputs_decomposed_softmax, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)

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

        for i in range(len(outputs_decomposed_original)):
            outputs_decomposed[i] = outputs_decomposed[i].detach().clone().requires_grad_(True)

    progress_bar.close()

    return (utility.applySoftmaxDecomposed(outputs_decomposed), None, None, None, None)

def counterfactual_pgd(outputs_decomposed_original, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, labels_original, device):
    batch_size = labels_original.nelement()
    outputs_decomposed = []
    outputs_decomposed_original = utility.applySoftmaxDecomposed(outputs_decomposed_original)
    progress_bar = tqdm.tqdm(total = batch_size, position = 1, leave = False)
    progress_bar.set_description_str("[INFO]: Optimizing attributes")

    for i in range(len(outputs_decomposed_original)):
        outputs_decomposed.append(outputs_decomposed_original[i].detach().clone().requires_grad_(True))

    for _ in range(header.counterfactual_steps):
        (_, _, outputs_composed_original) = utility.compose(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)
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

        for i in range(len(outputs_decomposed)):
            outputs_decomposed[i] = outputs_decomposed[i].detach().clone().requires_grad_(True)

    progress_bar.close()

    return (outputs_decomposed, None, None, None, None)

# Concatenate outputs_decomposed, outputs_decomposed_original, and y into one vector
# z = outputs_decomposed (x) - outputs_decomposed_original (x bar)
# z+ = max(0, z)
# z- = -min(0, z)
# outputs_decomposed (x) = z+ - z- + outputs_decomposed_original (x bar)
# Compute z+ and z- using QP and P, q, G, h, A, b

def counterfactual_pgd_qp(outputs_decomposed_original, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, labels_original, device):
    # Declare variables
    instance_count_done = False
    instance_count_qp_epsilon_violated = 0
    instance_count_qp_no_solution = 0
    instance_count_total = 0
    value_count_implausible = 0

    # Apply softmax to original decomposed outputs
    outputs_decomposed_original = utility.applySoftmaxDecomposed(outputs_decomposed_original)

    with torch.set_grad_enabled(False):
        # Initialize attribute sizes, attribute ranges, batch size, d, batched epsilon, and batched x bar
        attribute_sizes = []
        attribute_index = 0
        attribute_ranges = []
        batch_size = labels_original.nelement()
        x_bar_batch = []
        for k in range(len(outputs_decomposed_original)):
            attribute_sizes.append(outputs_decomposed_original[k].shape[1])
            attribute_range_left = attribute_index
            attribute_index += attribute_sizes[k]
            attribute_ranges.append((attribute_range_left, attribute_index))
            x_bar_batch.append(outputs_decomposed_original[k].detach().clone().requires_grad_(False).t())
        d = sum(attribute_sizes)    # sum of |A_i| for i in [1, K] where K is the number of attributes
        epsilon_batch = torch.full((1, batch_size), header.counterfactual_qp_epsilon).requires_grad_(False).to(device)   # 1 x batch size
        x_bar_batch = torch.cat(x_bar_batch, 0).requires_grad_(False)   # d x batch size

        # Initialize I_d, P, G, and batched h
        I_d = torch.eye(d).requires_grad_(False).to(device) # d x d
        P = torch.cat([torch.cat([ I_d, -I_d], 1).to(device),
                    torch.cat([-I_d,  I_d], 1).to(device)], 0).requires_grad_(False).to(device)  # 2d x 2d
        G = torch.cat([torch.cat([ -I_d, I_d], 1).to(device),
                    torch.ones(      1, 2 * d).to(device)], 0).requires_grad_(False).to(device)  # (d + 1) x 2d
        h_batch = torch.cat((x_bar_batch, epsilon_batch), 0).to(device)   # (d + 1) x batch size

        # Initialize A and batched b
        A_b_ones_list = []
        for k in range(len(attribute_sizes)):
            A_b_vector = torch.zeros(1, d).requires_grad_(False).to(device) # 1 x d
            A_b_vector[0][attribute_ranges[k][0]:attribute_ranges[k][1]] = 1
            A_b_ones_list.append(A_b_vector)
        A_b_ones = torch.cat(A_b_ones_list, 0).requires_grad_(False).to(device) # K x d
        A = torch.cat([A_b_ones, -1 * A_b_ones], 1).requires_grad_(False).to(device)    # K x 2d
        b_batch = 1 - torch.matmul(A_b_ones, x_bar_batch).requires_grad_(False).to(device)   # K x batch size

        # Log debug prints
        logger.log_trace("attribute_sizes:", attribute_sizes)
        logger.log_trace("attribute_ranges:", attribute_ranges)
        logger.log_trace("P:", P)
        logger.log_trace("P.shape:", P.shape)
        logger.log_trace("G:", G)
        logger.log_trace("G.shape:", G.shape)
        logger.log_trace("epsilon:", epsilon_batch)
        logger.log_trace("epsilon.shape:", epsilon_batch.shape)
        logger.log_trace("h:", h_batch)
        logger.log_trace("h.shape:", h_batch.shape)
        logger.log_trace("A:", A)
        logger.log_trace("A.shape:", A.shape)
        logger.log_trace("b_batch:", b_batch)
        logger.log_trace("b_batch.shape:", b_batch.shape)

    # Initialize gradients
    outputs_decomposed = []
    for k in range(len(outputs_decomposed_original)):
        outputs_decomposed.append(outputs_decomposed_original[k].detach().clone().requires_grad_(True))

    # Initialize progress bar
    progress_bar = tqdm.tqdm(total = batch_size, position = 1, leave = False)
    progress_bar.set_description_str("[INFO]: Optimizing attributes")

    # Enter optimization loop
    for _ in range(header.counterfactual_steps):
        # Compute composed output
        (_, _, outputs_composed_original) = utility.compose(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)

        # Obtain indices of output in the batch with unsatisfied validity and update progress bar
        outputs_composed_mpe = torch.max(outputs_composed_original, 1)[1]
        outputs_composed_indices = torch.where(outputs_composed_mpe != labels_original)[0]
        progress_bar.n = batch_size - outputs_composed_indices.nelement()
        progress_bar.refresh()

        # Break if validity of the entire batch is satisfied
        if outputs_composed_indices.nelement() == 0:
            break

        # Compute gradients
        outputs_composed = outputs_composed_original.t()   # number of original labels x batch size
        outputs_composed = outputs_composed[labels_original, torch.arange(outputs_composed.shape[1])]  # 1 x batch size
        outputs_composed = torch.log(outputs_composed)   # 1 x batch size
        outputs_composed = torch.sum(outputs_composed[outputs_composed_indices], 0) # 1 x 1
        outputs_composed.backward(retain_graph = True)

        with torch.set_grad_enabled(False):
            # Initialize batched x
            x_batch = []
            for k in range(len(outputs_decomposed)):
                x_batch.append(outputs_decomposed[k].detach().clone().requires_grad_(False).t())
            x_batch = torch.cat(x_batch, 0).requires_grad_(False).to(device) # d x batch size

            # Initialize batched y
            y_batch = []
            for k in range(len(outputs_decomposed)):
                y_batch.append((outputs_decomposed[k] + header.counterfactual_learning_rate * outputs_decomposed[k].grad).requires_grad_(False).t())
            y_batch = torch.cat(y_batch, 0).requires_grad_(False).to(device) # d x batch size

            # Initialize batched q
            q_batch = torch.cat([x_bar_batch - y_batch, -1 * (x_bar_batch - y_batch)], 0).requires_grad_(False).to(device)  # 2d x batch size

            # Initialize QP progress bar
            progress_bar_qp = tqdm.tqdm(total = batch_size, position = 2, leave = False)
            progress_bar_qp.set_description_str("[INFO]: Solving QP")

            # Solve QP for each batch
            z_plus_batch = []
            z_minus_batch = []
            for batch in range(batch_size):
                # Obtain column vectors for the current batch
                q = q_batch[:, batch].unsqueeze(0).t()  # 2d x 1
                h = h_batch[:, batch].unsqueeze(0).t()  # (d + 1) x 1
                b = b_batch[:, batch].unsqueeze(0).t()  # K x 1

                # Solve QP for z+ and z-
                z_plus_minus = qpsolvers.solve_qp(P.cpu().double().numpy(), q.cpu().double().numpy(), G.cpu().double().numpy(), h.cpu().double().numpy(), A.cpu().double().numpy(), b.cpu().double().numpy(), solver = header.counterfactual_qp_solver) # 2d x 1

                if z_plus_minus is None:
                    # Obtain column vectors for the current batch
                    x = x_batch[:, batch]   # d x 1
                    x_bar = x_bar_batch[:, batch]   # d x 1

                    # Obtain z+ and z- from unperturbed x
                    z = x - x_bar
                    z_plus = torch.clamp(z, min = 0).unsqueeze(0).requires_grad_(False).to(device).t()  # d x 1
                    z_minus = -1 * torch.clamp(z, max = 0).unsqueeze(0).requires_grad_(False).to(device).t()    # d x 1

                    # Count instances with no QP solutions
                    if not instance_count_done:
                        instance_count_qp_no_solution += 1
                else:
                    # Obtain z+ and z- from QP
                    z_plus = torch.from_numpy(z_plus_minus[:d]).unsqueeze(0).requires_grad_(False).to(device).t()   # d x 1
                    z_minus = torch.from_numpy(z_plus_minus[d:]).unsqueeze(0).requires_grad_(False).to(device).t()  # d x 1

                # Store z+ and z-
                z_plus_batch.append(z_plus)
                z_minus_batch.append(z_minus)

                # Count total instances
                if not instance_count_done:
                    instance_count_total += 1

                # Update QP progress bar
                progress_bar_qp.n = batch
                progress_bar_qp.refresh()

            # Close QP progress bar
            progress_bar_qp.close()

            # Initialize batched z+ and z-
            z_plus_batch = torch.cat(z_plus_batch, 1).requires_grad_(False).to(device)  # d x batch size
            z_minus_batch = torch.cat(z_minus_batch, 1).requires_grad_(False).to(device)    # d x batch size

            # Compute batched perturbed x and restore to decomposed outputs format
            x_batch_perturbed = z_plus_batch - z_minus_batch + x_bar_batch  # d x batch size
            outputs_decomposed_perturbed = torch.split(x_batch_perturbed.t(), attribute_sizes, 1)
            for k in range(len(outputs_decomposed)):
                outputs_decomposed[k] = outputs_decomposed_perturbed[k].float()

        # Initialize gradients
        for k in range(len(outputs_decomposed)):
            outputs_decomposed[k] = outputs_decomposed[k].detach().clone().requires_grad_(True)

        # Set instance count flag
        instance_count_done = True

    # Close progress bar
    progress_bar.close()

    # Disable gradients
    for k in range(len(outputs_decomposed)):
        outputs_decomposed[k] = outputs_decomposed[k].detach().clone().requires_grad_(False)

    # Compute L1 norm
    l1_norm = torch.zeros(batch_size).requires_grad_(False).to(device)  # batch size
    for (output_decomposed, output_decomposed_original) in zip(outputs_decomposed, outputs_decomposed_original):
        l1_norm += torch.norm((output_decomposed - output_decomposed_original), 1, 1).to(device).t()  # batch size

    # Count epsilon violations
    mask_qp_epsilon = (l1_norm > header.counterfactual_qp_epsilon)
    instance_count_qp_epsilon_violated = torch.sum(mask_qp_epsilon).item()

    # Restore perturbed outputs with L1-norm violations to unperturbed outputs
    for k in range(len(outputs_decomposed)):
        outputs_decomposed[k][mask_qp_epsilon, :] = outputs_decomposed_original[k][mask_qp_epsilon, :]

    # Count implausible values
    for k in range(len(outputs_decomposed)):
        # Range plausibility
        mask_lt_0 = (outputs_decomposed[k] < 0 - header.counterfactual_implausibility_margin_range)
        mask_gt_1 = (outputs_decomposed[k] > 1 + header.counterfactual_implausibility_margin_range)
        value_count_implausible += torch.sum(mask_lt_0 | mask_gt_1).item()

        # Sum plausibility
        output_decomposed_sum = torch.sum(outputs_decomposed[k], 1)
        mask_lt_1 = (output_decomposed_sum < 1 - header.counterfactual_implausibility_margin_sum)
        mask_gt_1 = (output_decomposed_sum > 1 + header.counterfactual_implausibility_margin_sum)
        value_count_implausible += torch.sum(mask_lt_1 | mask_gt_1).item()

    # Return perturbed decomposed outputs and instance counts
    return (outputs_decomposed, instance_count_qp_epsilon_violated, instance_count_qp_no_solution, instance_count_total, value_count_implausible)

def test(model_decomposed, spn_joint, spn_marginal, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)
    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])

    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    correctness_attribute_epoch = 0
    correctness_task_epoch = 0
    instance_count_corrected = 0
    instance_count_incorrect = 0
    instance_count_qp_epsilon_violated = None
    instance_count_qp_no_solution = 0
    instance_count_total = 0
    tv_distance_epoch = 0
    tv_distances_epoch = []
    value_count_implausible = None

    config_dataset = data_loader.dataset.config
    counterfactuals = {}
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    spn_output_rows = len(data_loader.dataset.classes_original)
    spn_output_cols = 1

    for attribute in config_dataset["attributes"]:
        spn_output_cols *= len(attribute["labels"])
        tv_distances_epoch.append(0)

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Counterfactual progress")

    for (batch_index, (input, labels_decomposed, labels_original, input_file_paths)) in enumerate(data_loader):
        input = input.to(device, non_blocking = True)
        labels_original = labels_original.to(device, non_blocking = True)

        for i in range(len(labels_decomposed)):
            labels_decomposed[i] = labels_decomposed[i].to(device, non_blocking = True)

        with torch.set_grad_enabled(False):
            (outputs_decomposed_original, _) = model_decomposed(input)
            outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed_original)

        with torch.set_grad_enabled(True):
            (outputs_decomposed_counterfactual, instance_count_batch_qp_epsilon_violated, instance_count_batch_qp_no_solution, instance_count_batch_total, value_count_batch_implausible) = counterfactual_pgd(outputs_decomposed_original, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, labels_original, device)

        if instance_count_batch_qp_epsilon_violated is not None:
            if instance_count_qp_epsilon_violated is None:
                instance_count_qp_epsilon_violated = 0

            instance_count_qp_epsilon_violated += instance_count_batch_qp_epsilon_violated

        if instance_count_batch_qp_no_solution is not None:
            instance_count_qp_no_solution += instance_count_batch_qp_no_solution

        if instance_count_batch_total is not None:
            instance_count_total += instance_count_batch_total

        if value_count_batch_implausible is not None:
            if value_count_implausible is None:
                value_count_implausible = 0

            value_count_implausible += value_count_batch_implausible

        (_, _, outputs_composed) = utility.compose(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)
        (_, _, outputs_composed_counterfactual) = utility.compose(outputs_decomposed_counterfactual, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)

        if header.counterfactual_save:
            utility.saveCounterfactuals(input_file_paths, counterfactuals, data_loader.dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_decomposed_counterfactual, outputs_composed, outputs_composed_counterfactual)

        (_, predictions_composed) = torch.max(outputs_composed, 1)
        (_, predictions_composed_counterfactual) = torch.max(outputs_composed_counterfactual, 1)

        corrects_composed = torch.sum(predictions_composed == labels_original).item()
        corrects_composed_counterfactual = torch.sum(predictions_composed_counterfactual == labels_original).item()
        incorrects_composed = torch.sum(predictions_composed != labels_original).item()

        instance_count_corrected += corrects_composed_counterfactual - corrects_composed
        instance_count_incorrect += incorrects_composed

        accuracy_attribute_epoch += utility.computeAccuracyDecomposed(outputs_decomposed_original, labels_decomposed, device)
        accuracy_task_epoch += corrects_composed
        correctness_attribute_epoch += utility.computeAccuracyDecomposed(outputs_decomposed_counterfactual, labels_decomposed, device)
        correctness_task_epoch += corrects_composed_counterfactual

        for i in range(len(data_loader.dataset.config["attributes"])):
            tv_distance_batch = 0.5 * torch.sum(torch.abs(outputs_decomposed_counterfactual[i] - outputs_decomposed[i]), dim = 1)
            tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    accuracy_task_epoch /= len(data_loader.dataset)
    correctness_attribute_epoch /= len(data_loader)
    correctness_task_epoch /= len(data_loader.dataset)

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distances_epoch[i] /= len(data_loader.dataset)

    tv_distance_epoch = sum(tv_distances_epoch) / len(tv_distances_epoch)

    logger.log_info("Testing attribute accuracy: " + str(accuracy_attribute_epoch) + ".")
    logger.log_info("Testing task accuracy: " + str(accuracy_task_epoch) + ".")

    if instance_count_incorrect != 0:
        logger.log_info("Correction rate: " + str(instance_count_corrected / instance_count_incorrect) + ".")
    else:
        logger.log_info("Correction rate: 0.")

    if instance_count_total != 0:
        logger.log_info("QP solution rate: " + str((instance_count_total - instance_count_qp_no_solution) / instance_count_total) + ".")

    if instance_count_qp_epsilon_violated is not None:
        logger.log_info("Number of epsilon violations: " + str(instance_count_qp_epsilon_violated) + ".")

    if value_count_implausible is not None:
        logger.log_info("Number of implausible values: " + str(value_count_implausible) + ".")

    logger.log_info("Counterfactual attribute TV distance: " + str(tv_distance_epoch) + ".")
    logger.log_info("Counterfactual attribute correctness: " + str(correctness_task_epoch) + ".")
    logger.log_info("Counterfactual task correctness: " + str(correctness_task_epoch) + ".")

    if not os.path.isdir(header.dir_output_counterfactual):
        os.makedirs(header.dir_output_counterfactual, exist_ok = True)

    with open(os.path.join(header.dir_output_counterfactual, header.file_name_counterfactual), "w") as file_counterfactual:
        json.dump(counterfactuals, file_counterfactual, indent = 4)

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

    test(model_decomposed, spn_joint, spn_marginal, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
