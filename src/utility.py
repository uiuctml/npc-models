import cv2
import logger
import natsort
import numpy
import os
import random
import torch
import torchvision
import wandb

def applySoftmax(output):
    return torch.nn.functional.softmax(output, dim = 1)

def applySoftmaxDecomposed(output_decomposed):
    output_decomposed_softmax = []

    for i in range(len(output_decomposed)):
        output_decomposed_softmax.append(applySoftmax(output_decomposed[i]))

    return output_decomposed_softmax

def compose(output_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device):
    log_likelihoods_joint = spn_joint.forward().to(device)
    log_likelihoods_marginal = spn_marginal.forward().to(device)

    # Compute matrix A and set entries with zero joint and marginal probabilities to zero
    mask_joint = (log_likelihoods_joint == -float("inf"))
    mask_marginal = (log_likelihoods_joint == -float("inf"))
    mask_matrix_a = mask_joint & mask_marginal
    matrix_a = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)
    matrix_a[mask_matrix_a] = 0
    matrix_a = matrix_a.reshape(spn_output_rows, spn_output_cols)

    batch_size = output_decomposed[0].shape[0]
    matrix_b_list = []

    for batch in range(batch_size):
        matrix_b_batch = output_decomposed[0][batch]

        for task_index in range(1, len(output_decomposed)):
            matrix_b_batch = torch.outer(matrix_b_batch, output_decomposed[task_index][batch]).flatten()

        matrix_b_list.append(matrix_b_batch)

    matrix_b = torch.stack(matrix_b_list, dim = 0).t()
    matrix_b = matrix_b.to(device)

    matrix_c = torch.matmul(matrix_a, matrix_b).t()

    return (matrix_a, matrix_b, matrix_c)

def composeSingle(batch, output_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device):
    log_likelihoods_joint = spn_joint.forward()
    log_likelihoods_marginal = spn_marginal.forward()

    # Compute matrix A and set entries with zero joint and marginal probabilities to zero
    mask_joint = (log_likelihoods_joint == -float("inf"))
    mask_marginal = (log_likelihoods_joint == -float("inf"))
    mask_matrix_a = mask_joint & mask_marginal
    matrix_a = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)
    matrix_a[mask_matrix_a] = 0
    matrix_a = matrix_a.reshape(spn_output_rows, spn_output_cols)

    matrix_b_batch = output_decomposed[0][batch]

    for task_index in range(1, len(output_decomposed)):
        matrix_b_batch = torch.outer(matrix_b_batch, output_decomposed[task_index][batch]).flatten()

    matrix_b_batch = matrix_b_batch.unsqueeze(0).to(device).t()
    matrix_c = torch.matmul(matrix_a, matrix_b_batch).t().squeeze(0)

    return (matrix_a, matrix_b_batch, matrix_c)

def computeAccuracyDecomposed(output, labels, device):
    count_attributes = len(output)
    corrects = []

    for i in range(count_attributes):
        masks_labels = (labels[i] > 0)
        counts_values = torch.sum(masks_labels, dim = 1)
        masks_predictions = []

        for (output_batch, count_value_batch) in zip(output[i], counts_values):
            mask_prediction_batch = torch.zeros(output_batch.shape, dtype = torch.bool)
            (_, prediction_batch) = torch.topk(output_batch, count_value_batch)
            mask_prediction_batch[prediction_batch] = True
            masks_predictions.append(mask_prediction_batch)

        masks_predictions = torch.stack(masks_predictions, dim = 0).to(device)
        corrects_attribute = torch.all(masks_predictions == masks_labels, dim = 1)

        corrects.append(corrects_attribute)

    corrects = torch.stack(corrects, dim = 1)
    corrects = torch.all(corrects, dim = 1)
    accuracy = (torch.sum(corrects) / corrects.size(0)).item()

    return accuracy

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

def createTransform(config):
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config["model_input_height"], config["model_input_width"])),
        torchvision.transforms.ToTensor()
    ])

    return dataset_transforms

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

def getBinaryLabelsDecomposed(labels_decomposed):
    labels_decomposed_binary = []

    for label_decomposed in labels_decomposed:
        labels_decomposed_binary.append((label_decomposed > 0).float())

    return torch.cat(labels_decomposed_binary, dim = 1)

def getBinaryLabelsOriginal(labels_original, data_loader):
    return torch.nn.functional.one_hot(labels_original, num_classes = len(data_loader.dataset.classes_original)).float()

def getLabelsAttribute(dataset_config):
    labels_attribute = {}

    for attribute in dataset_config["attributes"]:
        if "" in attribute["labels"]:
            attribute["labels"].remove("")

        labels_attribute[attribute["name"]] = attribute["labels"]

    return labels_attribute

def getLabelsOriginal(dataset_config):
    if "instance_wise" in dataset_config and dataset_config["instance_wise"]:
        labels_original = []
        labels_original_set = set()

        for image_name in dataset_config["mappings"].keys():
            class_name = image_name.split('/')[0]

            if class_name not in labels_original_set:
                labels_original.append(class_name)
                labels_original_set.add(class_name)

        return natsort.natsorted(labels_original)
    else:
        return list(dataset_config["mappings"].keys())

def getIndicesFromLabelsAttribute(labels_attribute):
    indices = {}

    for attribute in labels_attribute.keys():
        labels_to_indices = {}

        for i in range(len(labels_attribute[attribute])):
            labels_to_indices[labels_attribute[attribute][i]] = i

        indices[attribute] = labels_to_indices

    return indices

def getIndicesFromLabelsOriginal(labels_original):
    labels_to_indices = {}

    for i in range(len(labels_original)):
        labels_to_indices[labels_original[i]] = i

    return labels_to_indices

def getMatrixAColIndicesAttributeIndicesMaps(matrix_a_cols, spn_settings):
    attribute_indices_list = spn_settings[:matrix_a_cols, :-1].cpu().int().tolist()
    matrix_a_col_indices_to_attribute_indices = {}
    attribute_indices_to_matrix_a_col_indices = {}

    for (matrix_a_row_index, attribute_indices) in enumerate(attribute_indices_list):
        matrix_a_col_indices_to_attribute_indices[matrix_a_row_index] = tuple(attribute_indices)
        attribute_indices_to_matrix_a_col_indices[tuple(attribute_indices)] = matrix_a_row_index

    return (matrix_a_col_indices_to_attribute_indices, attribute_indices_to_matrix_a_col_indices)

def generateSPNSettings(config_dataset, device):
    attribute_ranges = []
    labels_attribute = getLabelsAttribute(config_dataset)
    labels_original = getLabelsOriginal(config_dataset)

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

def loadCheckpoint(dir_checkpoints, file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_schedulers, model, optimizers):
    if wandb.run.resumed:
        if not os.path.isdir(dir_checkpoints):
            os.makedirs(dir_checkpoints, exist_ok = True)

        try:
            wandb.restore(file_name_checkpoint, root = dir_checkpoints)
        except:
            pass
        else:
            logger.log_info("Restored checkpoint \"" + file_name_checkpoint + "\" from Weights & Biases.")

        file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

        if os.path.isfile(file_path_checkpoint):
            checkpoint = torch.load(file_path_checkpoint)
            accuracy_validation_best = checkpoint["accuracy_validation_best"]
            batch_step_train = checkpoint["batch_step_train"]
            batch_step_validate = checkpoint["batch_step_validate"]
            criterions = checkpoint["criterions"]
            epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])

            for (learning_rate_scheduler, learning_rate_scheduler_state_dict) in zip(learning_rate_schedulers, checkpoint["learning_rate_scheduler_state_dict_list"]):
                learning_rate_scheduler.load_state_dict(learning_rate_scheduler_state_dict)

                if (isinstance(learning_rate_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR) or isinstance(learning_rate_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)):
                    learning_rate_scheduler.last_epoch = epoch

            for (optimizer, optimizer_state_dict) in zip(optimizers, checkpoint["optimizer_state_dict_list"]):
                optimizer.load_state_dict(optimizer_state_dict)

            logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
        else:
            logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
            exit(-1)

    return (accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch)

def loadCheckpointBest(dir_checkpoints, file_name_checkpoint, model):
    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok = True)

    try:
        wandb.restore(file_name_checkpoint, root = dir_checkpoints)
    except:
        pass
    else:
        logger.log_info("Restored checkpoint \"" + file_name_checkpoint + "\" from Weights & Biases.")

    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    if os.path.isfile(file_path_checkpoint):
        checkpoint = torch.load(file_path_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
    else:
        logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
        exit(-1)

    return

def loadCheckpointBestSPN(spn, dir_checkpoints, file_name_checkpoint):
    if not os.path.isdir(dir_checkpoints):
        logger.log_fatal("Checkpoint directory \"" + dir_checkpoints + "\" missing.")
        exit(-1)

    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    if os.path.isfile(file_path_checkpoint):
        checkpoint = torch.load(file_path_checkpoint)
        spn.set_weights(checkpoint["weights"])

        logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
    else:
        logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
        exit(-1)

    return

def loadCheckpointSPN(spn, dir_checkpoints, file_name_checkpoint, log_likelihood_best, log_likelihood_train_last, epoch):
    if wandb.run.resumed:
        if not os.path.isdir(dir_checkpoints):
            logger.log_fatal("Checkpoint directory \"" + dir_checkpoints + "\" missing.")
            exit(-1)

        try:
            wandb.restore(file_name_checkpoint, root = dir_checkpoints)
        except:
            pass
        else:
            logger.log_info("Restored checkpoint \"" + file_name_checkpoint + "\" from Weights & Biases.")

        file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

        if os.path.isfile(file_path_checkpoint):
            checkpoint = torch.load(file_path_checkpoint)
            epoch = checkpoint["epoch"]
            log_likelihood_best = checkpoint["log_likelihood_best"]
            log_likelihood_train_last = checkpoint["log_likelihood_train_last"]
            spn.set_weights(checkpoint["weights"])

            logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
        else:
            logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
            exit(-1)

    return (log_likelihood_best, log_likelihood_train_last, epoch)

def lossNegativeLogLikelihood(output, label):
    label = label.reshape(-1, 1)
    output = torch.gather(output, 1, label)
    negative_log_likelihood = -1 * torch.log(output)

    return negative_log_likelihood.mean()

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    width_image = image.shape[1]
    width_resize = image.shape[1]
    height_image = image.shape[0]
    height_resize = image.shape[0]

    if width is None and height is None:
        return image

    if width is None:
        resize_ratio = height / height_image
        width_resize = int(width_image * resize_ratio)
        height_resize = height
    else:
        resize_ratio = width / width_image
        width_resize = width
        height_resize = int(height_image * resize_ratio)

    return cv2.resize(image, (width_resize, height_resize), interpolation=inter)

def saveCheckpoint(dir_checkpoints, file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_schedulers, model, optimizers):
    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok = True)

    learning_rate_scheduler_state_dict_list = []
    optimizer_state_dict_list = []

    for learning_rate_scheduler in learning_rate_schedulers:
        learning_rate_scheduler_state_dict_list.append(learning_rate_scheduler.state_dict())

    for optimizer in optimizers:
        optimizer_state_dict_list.append(optimizer.state_dict())

    checkpoint = {
        "accuracy_validation_best": accuracy_validation_best,
        "batch_step_train": batch_step_train,
        "batch_step_validate": batch_step_validate,
        "criterions": criterions,
        "epoch": epoch,
        "learning_rate_scheduler_state_dict_list": learning_rate_scheduler_state_dict_list,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict_list": optimizer_state_dict_list
    }
    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    torch.save(checkpoint, file_path_checkpoint)

    try:
        wandb.save(file_path_checkpoint, base_path = dir_checkpoints)
    except:
        pass
    else:
        logger.log_trace("Saved checkpoint \"" + file_name_checkpoint + "\" to Weights & Biases.")

    logger.log_trace("Saved checkpoint \"" + file_name_checkpoint + "\".")

    return

def saveCheckpointSPN(spn, dir_checkpoints, file_name_checkpoint, log_likelihood_best, log_likelihood_train_last, epoch):
    weights = spn.get_weights()

    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok = True)

    checkpoint = {
        "epoch": epoch,
        "log_likelihood_best": log_likelihood_best,
        "log_likelihood_train_last": log_likelihood_train_last,
        "weights": weights
    }
    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    torch.save(checkpoint, file_path_checkpoint)

    try:
        wandb.save(file_path_checkpoint, base_path = dir_checkpoints)
    except:
        pass
    else:
        logger.log_trace("Saved checkpoint \"" + file_name_checkpoint + "\" to Weights & Biases.")

    logger.log_trace("Saved checkpoint \"" + file_name_checkpoint + "\".")

    return

def saveCounterfactuals(input_file_paths, counterfactuals, dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_decomposed_counterfactual, outputs_composed, outputs_composed_counterfactual):
    batch_size = labels_original.nelement()
    config_dataset = dataset.config

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        counterfactuals[input_file_path] = {}
        counterfactuals[input_file_path]["counterfactual"] = {}
        counterfactuals[input_file_path]["ground_truth"] = {}
        counterfactuals[input_file_path]["prediction"] = {}

        for (attribute_index, attribute) in enumerate(config_dataset["attributes"]):
            attribute_name = attribute["name"]

            (outputs_decomposed_counterfactual_mpe_probability, outputs_decomposed_counterfactual_mpe_label) = torch.max(outputs_decomposed_counterfactual[attribute_index][batch_index], 0)
            (outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label) = torch.max(outputs_decomposed[attribute_index][batch_index], 0)

            counterfactuals[input_file_path]["counterfactual"][attribute_name] = (dataset.classes[attribute_name][outputs_decomposed_counterfactual_mpe_label], outputs_decomposed_counterfactual_mpe_probability.item())
            counterfactuals[input_file_path]["ground_truth"][attribute_name] = dataset.classes[attribute_name][labels_decomposed[batch_index][attribute_index]]
            counterfactuals[input_file_path]["prediction"][attribute_name] = (dataset.classes[attribute_name][outputs_decomposed_mpe_label], outputs_decomposed_mpe_probability.item())

        (outputs_composed_counterfactual_mpe_probability, outputs_composed_counterfactual_mpe_label) = torch.max(outputs_composed_counterfactual[batch_index], 0)
        (outputs_composed_mpe_probability, outputs_composed_mpe_label) = torch.max(outputs_composed[batch_index], 0)

        counterfactuals[input_file_path]["counterfactual"]["original"] = (dataset.classes_original[outputs_composed_counterfactual_mpe_label], outputs_composed_counterfactual_mpe_probability.item())
        counterfactuals[input_file_path]["ground_truth"]["original"] = dataset.classes_original[labels_original[batch_index]]
        counterfactuals[input_file_path]["prediction"]["original"] = (dataset.classes_original[outputs_composed_mpe_label], outputs_composed_mpe_probability.item())

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

def setSeed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    return

def wAndBDefineMetrics():
    wandb.define_metric("testing/batch/step")
    wandb.define_metric("testing/epoch/step")
    wandb.define_metric("training/batch/step")
    wandb.define_metric("training/epoch/step")
    wandb.define_metric("validation/batch/step")
    wandb.define_metric("validation/epoch/step")

    wandb.define_metric("testing/batch/*", step_metric = "testing/batch/step")
    wandb.define_metric("testing/epoch/*", step_metric = "testing/epoch/step")
    wandb.define_metric("training/batch/*", step_metric = "training/batch/step")
    wandb.define_metric("training/epoch/*", step_metric = "training/epoch/step")
    wandb.define_metric("validation/batch/*", step_metric = "validation/batch/step")
    wandb.define_metric("validation/epoch/*", step_metric = "validation/epoch/step")

    return
