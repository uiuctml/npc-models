#!/usr/bin/env python3

import argparse
import dataset
import header
import logger
import model
import spn
import test_composed
import test_dl
import test_spn
import torch
import tqdm
import utility
import wandb

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights-attribute", type = str, default = "", help = "Attribute model pretrained weights.")
    parser.add_argument("-c", "--weights-pc", type = str, default = "", help = "PC model pretrained weights.")
    parser.add_argument("-b", "--batch-size", type = int, default = None, help = "Batch size.")
    parser.add_argument("-e", "--epochs", type = int, default = None, help = "Epochs.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Random seed.")
    arguments = parser.parse_args()

    test_dl.initializeRunName()
    test_spn.initializeRunName("", "pgd")

    if arguments.weights_attribute != "":
        header.config_decomposed["model_pretrained_weights"] = arguments.weights_attribute

    if arguments.weights_pc != "":
        header.config_spn["model_pretrained_weights"] = arguments.weights_pc

    if arguments.batch_size is not None:
        header.config_decomposed["batch_size"] = arguments.batch_size

    if arguments.epochs is not None:
        header.config_decomposed["epochs"] = arguments.epochs

    if arguments.seed is not None:
        header.config_decomposed["seed"] = arguments.seed

    logger.log_trace("Attribute run name: \"" + header.config_decomposed["run_name"] + "\".")
    logger.log_trace("PC run name: \"" + header.config_spn["run_name"] + "\".")
    logger.log_trace("Attribute model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
    logger.log_trace("PC model pretrained weights: \"" + header.config_spn["model_pretrained_weights"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_decomposed["batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_decomposed["epochs"]) + ".")
    logger.log_trace("Random seed: " + str(header.config_decomposed["seed"]) + ".")

    return

def train(model_decomposed, spn_joint, spn_marginal, data_loader, criterion, optimizer_decomposed, optimizer_spn, device, batch_step):
    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)
    spn_output_rows = len(data_loader.dataset.classes_original)
    spn_output_cols = 1

    for attribute in data_loader.dataset.config["attributes"]:
        attribute_labels = attribute["labels"]
        spn_output_cols *= len(attribute_labels)

    model_decomposed.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    with torch.set_grad_enabled(True):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            for i in range(len(labels_decomposed)):
                labels_decomposed[i] = labels_decomposed[i].to(device, non_blocking = True)

            optimizer_decomposed.zero_grad()

            (output_decomposed, _) = model_decomposed(input)

            output_decomposed = utility.applySoftmaxDecomposed(output_decomposed)
            (matrix_a, matrix_b, output_composed) = utility.compose(output_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)

            (_, predictions_composed) = torch.max(output_composed, 1)
            loss = criterion(output_composed, labels_original)

            loss.backward()
            optimizer_decomposed.step()

            if not header.config_spn["joint_inference_only"]:
                spn_joint.backward()
                spn_marginal.backward()

                optimizer_spn.step(matrix_a.detach(), matrix_b.detach(), output_composed.detach(), labels_original)

            corrects_composed = torch.sum(predictions_composed == labels_original).item()

            accuracy_attribute_batch = utility.computeAccuracyDecomposed(output_decomposed, labels_decomposed, device)
            accuracy_task_batch = corrects_composed / input.size(0)
            loss_batch = loss.item()

            accuracy_attribute_epoch += accuracy_attribute_batch
            accuracy_task_epoch += corrects_composed
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"training/batch/accuracy_attribute": accuracy_attribute_batch})
            wandb.log({"training/batch/accuracy_task": accuracy_task_batch})
            wandb.log({"training/batch/step": batch_step})
            wandb.log({"training/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    accuracy_task_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"training/epoch/accuracy_attribute": accuracy_attribute_epoch})
    wandb.log({"training/epoch/accuracy_task": accuracy_task_epoch})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def validate(model_decomposed, spn_joint, spn_marginal, data_loader, criterion, device, batch_step):
    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)
    spn_output_rows = len(data_loader.dataset.classes_original)
    spn_output_cols = 1

    for attribute in data_loader.dataset.config["attributes"]:
        attribute_labels = attribute["labels"]
        spn_output_cols *= len(attribute_labels)

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            for i in range(len(labels_decomposed)):
                labels_decomposed[i] = labels_decomposed[i].to(device, non_blocking = True)

            (output_decomposed, _) = model_decomposed(input)

            output_decomposed = utility.applySoftmaxDecomposed(output_decomposed)
            (_, _, output_composed) = utility.compose(output_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device)

            (_, predictions_composed) = torch.max(output_composed, 1)
            loss = criterion(output_composed, labels_original)

            corrects_composed = torch.sum(predictions_composed == labels_original).item()

            accuracy_attribute_batch = utility.computeAccuracyDecomposed(output_decomposed, labels_decomposed, device)
            accuracy_task_batch = corrects_composed / input.size(0)
            loss_batch = loss.item()

            accuracy_attribute_epoch += accuracy_attribute_batch
            accuracy_task_epoch += corrects_composed
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"validation/batch/accuracy_attribute": accuracy_attribute_batch})
            wandb.log({"validation/batch/accuracy_task": accuracy_task_batch})
            wandb.log({"validation/batch/step": batch_step})
            wandb.log({"validation/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    accuracy_task_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"validation/epoch/accuracy_attribute": accuracy_attribute_epoch})
    wandb.log({"validation/epoch/accuracy_task": accuracy_task_epoch})
    wandb.log({"validation/epoch/loss": loss_epoch})

    return (accuracy_task_epoch, loss_epoch, batch_step)

def main():
    processArguments()

    utility.setSeed(header.config_decomposed["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    config = {
        "decomposed": header.config_decomposed,
        "spn": header.config_spn
    }

    wandb.init(project = header.project_name, name = header.config_decomposed["run_name"], config = config, mode = header.run_mode)
    utility.defineMetrics()
    logger.log_info("Started run \"" + header.config_decomposed["run_name"] + "\" and " + header.config_spn["run_name"] + ".")

    accuracy_task_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterion = utility.lossNegativeLogLikelihood
    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.NPCDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.NPCDataset(header.config_decomposed["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.NPCDataset(header.config_decomposed["dir_dataset_validation"], dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_decomposed["batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_decomposed["batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    device_spn = torch.device("cuda")

    if header.composed_spn_on_cpu:
        logger.log_info("Computing SPNs on CPU.")
        device_spn = torch.device("cpu")

    epoch = 1
    model_decomposed = model.ResNet34MTL(dataset_test.config, device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    progress_bar = None
    spn_joint = spn.SPN(device_spn)
    spn_marginal = spn.SPN(device_spn)
    optimizer_decomposed = torch.optim.SGD(model_decomposed.parameters(), lr = header.config_decomposed["optimizer_learning_rate"], momentum = header.config_decomposed["optimizer_momentum"], weight_decay = header.config_decomposed["optimizer_weight_decay"])
    optimizer_spn = optimizer_spn = spn.PGDDiscriminativeSPNOptimizer(spn_joint, spn_marginal, device_spn, header.config_spn["optimizer_learning_rate"], header.config_spn["optimizer_prior_factor"], header.config_spn["epsilon_projection"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_decomposed, header.config_decomposed["learning_rate_scheduler_mode"], header.config_decomposed["learning_rate_scheduler_factor"], header.config_decomposed["learning_rate_scheduler_patience"], header.config_decomposed["learning_rate_scheduler_threshold"], header.config_decomposed["learning_rate_scheduler_threshold_mode"], header.config_decomposed["learning_rate_scheduler_cooldown"], header.config_decomposed["learning_rate_scheduler_min_learning_rate"], header.config_decomposed["learning_rate_scheduler_min_learning_rate_decay"])
    learning_rate_scheduler_spn = spn.LossSPNLearningRateScheduler(optimizer_spn, header.config_spn["learning_rate_scheduler_factor"], header.config_spn["learning_rate_scheduler_patience"], header.config_spn["learning_rate_scheduler_threshold"], header.config_spn["learning_rate_scheduler_cooldown"], header.config_spn["learning_rate_scheduler_min_learning_rate"])

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])
    optimizer_spn.set_weights_prior(spn_joint.get_weights())

    logger.log_info("Loading SPN leaf node settings...")

    spn_settings_joint = utility.generateSPNSettings(config_dataset, device)
    spn_settings_marginal = torch.clone(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    logger.log_info("Setting SPN leaf nodes...")

    spn_joint.set_leaf_nodes_categorical(spn_settings_joint)
    spn_marginal.set_leaf_nodes_categorical(spn_settings_marginal)

    if header.config_decomposed["model_pretrained_weights"] != "":
        utility.loadCheckpoint(header.config_decomposed["model_pretrained_weights"], model_decomposed)

    if header.config_spn["model_pretrained_weights"] != "":
        utility.loadCheckpoint(header.config_spn["model_pretrained_weights"], spn_joint, True)
        utility.loadCheckpoint(header.config_spn["model_pretrained_weights"], spn_marginal, True)
    elif header.config_spn["randomize_weights"]:
        logger.log_info("Randomizing SPN weights...")
        spn_joint.randomize_weights()
        spn_marginal.set_weights(spn_joint.get_weights())

    logger.log_trace("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
    logger.log_trace("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
    logger.log_trace("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
    logger.log_trace("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
    logger.log_trace("SPN depth: " + str(spn_joint.depth) + ".")
    logger.log_trace("SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    if epoch <= header.config_decomposed["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_decomposed["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_decomposed["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_decomposed, spn_joint, spn_marginal, data_loader_train, criterion, optimizer_decomposed, optimizer_spn, device, batch_step_train)
        (accuracy_task_validation_epoch, loss_validation_epoch, batch_step_validate) = validate(model_decomposed, spn_joint, spn_marginal, data_loader_validation, criterion, device, batch_step_validate)

        learning_rate_scheduler.step(loss_validation_epoch)
        learning_rate_scheduler_spn.step(loss_validation_epoch)

        logger.log_info("Epoch validation task accuracy: " + str(accuracy_task_validation_epoch) + ".")

        if accuracy_task_validation_epoch > accuracy_task_validation_best or epoch == 1:
            accuracy_task_validation_best = accuracy_task_validation_epoch
            wandb.log({"validation/epoch/accuracy_task_best": accuracy_task_validation_best})
            utility.saveCheckpoint(header.config_decomposed["file_name_checkpoint_best"], model_decomposed)
            utility.saveCheckpoint(header.config_spn["file_name_checkpoint_best"], spn_joint, True)

        utility.saveCheckpoint(header.config_decomposed["file_name_checkpoint"], model_decomposed)
        utility.saveCheckpoint(header.config_spn["file_name_checkpoint"], spn_joint, True)

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation task accuracy: " + str(accuracy_task_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_task_best"] = accuracy_task_validation_best

    if not header.config_spn["joint_inference_only"]:
        settings_marginal = torch.full((1, spn_settings_joint.shape[1]), -1).to(device)
        logger.log_info("Normalizing SPN weights...")

        utility.loadCheckpoint(header.config_spn["file_name_checkpoint"], spn_marginal, True)
        spn_marginal(settings_marginal)
        spn_marginal.normalize_weights(header.config_spn["epsilon_smoothing"])
        utility.saveCheckpoint(header.config_spn["file_name_checkpoint"], spn_marginal, True)

        utility.loadCheckpoint(header.config_spn["file_name_checkpoint_best"], spn_marginal, True)
        spn_marginal(settings_marginal)
        spn_marginal.normalize_weights(header.config_spn["epsilon_smoothing"])
        utility.saveCheckpoint(header.config_spn["file_name_checkpoint_best"], spn_marginal, True)

        spn_marginal.set_leaf_nodes_categorical(spn_settings_marginal)

    wandb.log({"testing/epoch/step": batch_step_test})
    test_composed.test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
