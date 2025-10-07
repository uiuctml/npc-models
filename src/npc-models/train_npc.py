#!/usr/bin/env python3

import argparse
import dataset
import header
import logger
import model
import pc
import test_neural
import test_npc
import test_pc
import torch
import tqdm
import utility
import wandb

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights-neural", type = str, default = "", help = "Neural model pretrained weights.")
    parser.add_argument("-c", "--weights-pc", type = str, default = "", help = "PC model pretrained weights.")
    parser.add_argument("-b", "--batch-size", type = int, default = None, help = "Batch size.")
    parser.add_argument("-e", "--epochs", type = int, default = None, help = "Epochs.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Seed.")
    arguments = parser.parse_args()

    test_neural.initializeRunName()
    test_pc.initializeRunName("", "pgd")

    if arguments.weights_neural != "":
        header.config_neural["model_pretrained_weights"] = arguments.weights_neural

    if arguments.weights_pc != "":
        header.config_pc["model_pretrained_weights"] = arguments.weights_pc

    if arguments.batch_size is not None:
        header.config_neural["batch_size"] = arguments.batch_size

    if arguments.epochs is not None:
        header.config_neural["epochs"] = arguments.epochs

    if arguments.seed is not None:
        header.config_neural["seed"] = arguments.seed

    logger.log_trace("Neural run name: \"" + header.config_neural["run_name"] + "\".")
    logger.log_trace("Neural model pretrained weights: \"" + header.config_neural["model_pretrained_weights"] + "\".")
    logger.log_trace("PC run name: \"" + header.config_pc["run_name"] + "\".")
    logger.log_trace("PC model pretrained weights: \"" + header.config_pc["model_pretrained_weights"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_neural["batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_neural["epochs"]) + ".")
    logger.log_trace("Seed: " + str(header.config_neural["seed"]) + ".")

    return

def train(model_neural, pc_joint, pc_marginal, data_loader, optimizer_neural, optimizer_pc, device, batch_step):
    accuracy_concept_epoch = 0
    accuracy_classification_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)
    pc_output_rows = len(data_loader.dataset.labels_class)
    pc_output_cols = 1

    for attribute in data_loader.dataset.config["attributes"]:
        attribute_labels = attribute["labels"]
        pc_output_cols *= len(attribute_labels)

    model_neural.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    with torch.set_grad_enabled(True):
        for (batch_index, (input, labels_attribute, labels_class, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_class = labels_class.to(device, non_blocking = True)

            for i in range(len(labels_attribute)):
                labels_attribute[i] = labels_attribute[i].to(device, non_blocking = True)

            optimizer_neural.zero_grad()

            (outputs_attribute, _) = model_neural(input)

            outputs_attribute = utility.applySoftmaxAttribute(outputs_attribute)
            (matrix_pc, matrix_neural, output_npc) = test_npc.computeNPCOutput(outputs_attribute, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)

            (_, predictions_npc) = torch.max(output_npc, 1)
            loss = utility.lossNegativeLogLikelihood(output_npc, labels_class)

            loss.backward()
            optimizer_neural.step()

            if header.npc_pc_backward:
                pc_joint.backward()
                pc_marginal.backward()

                optimizer_pc.step(matrix_pc.detach(), matrix_neural.detach(), output_npc.detach(), labels_class)

            corrects_npc = torch.sum(predictions_npc == labels_class).item()

            accuracy_concept_batch = utility.computeConceptAccuracy(outputs_attribute, labels_attribute, device)
            accuracy_classification_batch = corrects_npc / input.size(0)
            loss_batch = loss.item()

            accuracy_concept_epoch += accuracy_concept_batch
            accuracy_classification_epoch += corrects_npc
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"training/batch/accuracy_concept": accuracy_concept_batch})
            wandb.log({"training/batch/accuracy_classification": accuracy_classification_batch})
            wandb.log({"training/batch/step": batch_step})
            wandb.log({"training/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_concept_epoch /= len(data_loader)
    accuracy_classification_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"training/epoch/accuracy_concept": accuracy_concept_epoch})
    wandb.log({"training/epoch/accuracy_classification": accuracy_classification_epoch})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def validate(model_neural, pc_joint, pc_marginal, data_loader, device, batch_step):
    accuracy_concept_epoch = 0
    accuracy_classification_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)
    pc_output_rows = len(data_loader.dataset.labels_class)
    pc_output_cols = 1

    for attribute in data_loader.dataset.config["attributes"]:
        attribute_labels = attribute["labels"]
        pc_output_cols *= len(attribute_labels)

    model_neural.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_attribute, labels_class, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_class = labels_class.to(device, non_blocking = True)

            for i in range(len(labels_attribute)):
                labels_attribute[i] = labels_attribute[i].to(device, non_blocking = True)

            (outputs_attribute, _) = model_neural(input)

            outputs_attribute = utility.applySoftmaxAttribute(outputs_attribute)
            (_, _, output_npc) = test_npc.computeNPCOutput(outputs_attribute, pc_joint, pc_marginal, pc_output_rows, pc_output_cols, device)

            (_, predictions_npc) = torch.max(output_npc, 1)
            loss = utility.lossNegativeLogLikelihood(output_npc, labels_class)

            corrects_npc = torch.sum(predictions_npc == labels_class).item()

            accuracy_concept_batch = utility.computeConceptAccuracy(outputs_attribute, labels_attribute, device)
            accuracy_classification_batch = corrects_npc / input.size(0)
            loss_batch = loss.item()

            accuracy_concept_epoch += accuracy_concept_batch
            accuracy_classification_epoch += corrects_npc
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"validation/batch/accuracy_concept": accuracy_concept_batch})
            wandb.log({"validation/batch/accuracy_classification": accuracy_classification_batch})
            wandb.log({"validation/batch/step": batch_step})
            wandb.log({"validation/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_concept_epoch /= len(data_loader)
    accuracy_classification_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"validation/epoch/accuracy_concept": accuracy_concept_epoch})
    wandb.log({"validation/epoch/accuracy_classification": accuracy_classification_epoch})
    wandb.log({"validation/epoch/loss": loss_epoch})

    return (accuracy_classification_epoch, loss_epoch, batch_step)

def main():
    processArguments()

    utility.setSeed(header.config_neural["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    config = {
        "neural": header.config_neural,
        "pc": header.config_pc
    }

    wandb.init(project = header.project_name, name = header.config_neural["run_name"], config = config, mode = header.run_mode)
    utility.defineMetrics()
    logger.log_info("Started run \"" + header.config_neural["run_name"] + "\" and " + header.config_pc["run_name"] + ".")

    accuracy_classification_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    dataset_transforms = utility.createTransforms(header.config_neural)
    dataset_test = dataset.NPCDataset(header.config_neural["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.NPCDataset(header.config_neural["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.NPCDataset(header.config_neural["dir_dataset_validation"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_neural["batch_size"], shuffle = False, num_workers = header.config_neural["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_neural["batch_size"], shuffle = header.config_neural["data_loader_shuffle"], num_workers = header.config_neural["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_neural["batch_size"], shuffle = header.config_neural["data_loader_shuffle"], num_workers = header.config_neural["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    device_pc = torch.device("cuda")

    if header.npc_pc_cpu:
        logger.log_info("Computing PCs on CPU.")
        device_pc = torch.device("cpu")

    epoch = 1
    model_neural = model.ResNet34MTL(dataset_test.config, device)
    model_neural = torch.nn.DataParallel(model_neural)
    model_neural = model_neural.to(device)
    pc_joint = pc.ProbabilisticCircuit(device_pc)
    pc_marginal = pc.ProbabilisticCircuit(device_pc)
    optimizer_neural = torch.optim.SGD(model_neural.parameters(), lr = header.config_neural["optimizer_learning_rate"], momentum = header.config_neural["optimizer_momentum"], weight_decay = header.config_neural["optimizer_weight_decay"])
    optimizer_pc = pc.PGDPCOptimizer(pc_joint, pc_marginal, device_pc, header.config_pc["optimizer_learning_rate"], header.config_pc["optimizer_prior_factor"], header.config_pc["epsilon_projection"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_neural, header.config_neural["learning_rate_scheduler_mode"], header.config_neural["learning_rate_scheduler_factor"], header.config_neural["learning_rate_scheduler_patience"], header.config_neural["learning_rate_scheduler_threshold"], header.config_neural["learning_rate_scheduler_threshold_mode"], header.config_neural["learning_rate_scheduler_cooldown"], header.config_neural["learning_rate_scheduler_min_learning_rate"], header.config_neural["learning_rate_scheduler_min_learning_rate_decay"])
    learning_rate_scheduler_pc = pc.LossPCLearningRateScheduler(optimizer_pc, header.config_pc["learning_rate_scheduler_factor"], header.config_pc["learning_rate_scheduler_patience"], header.config_pc["learning_rate_scheduler_threshold"], header.config_pc["learning_rate_scheduler_cooldown"], header.config_pc["learning_rate_scheduler_min_learning_rate"])

    logger.log_info("Loading PC \"" + header.config_pc["file_path_pc"] + "\"...")

    pc_joint.load(header.config_pc["file_path_pc"])
    pc_marginal.load(header.config_pc["file_path_pc"])
    optimizer_pc.set_weights_prior(pc_joint.get_weights())

    logger.log_info("Loading PC leaf node settings...")

    pc_settings_joint = test_npc.generatePCSettings(dataset_test.config, device)
    pc_settings_marginal = torch.clone(pc_settings_joint)
    pc_settings_marginal[:, -1] = -1

    logger.log_info("Setting PC leaf nodes...")

    pc_joint.set_leaf_nodes_categorical(pc_settings_joint)
    pc_marginal.set_leaf_nodes_categorical(pc_settings_marginal)

    if header.config_neural["model_pretrained_weights"] != "":
        utility.loadCheckpoint(header.config_neural["model_pretrained_weights"], model_neural)

    if header.config_pc["model_pretrained_weights"] != "":
        utility.loadCheckpoint(header.config_pc["model_pretrained_weights"], pc_joint, True)
        utility.loadCheckpoint(header.config_pc["model_pretrained_weights"], pc_marginal, True)
    elif header.config_pc["randomize_weights"]:
        logger.log_info("Randomizing PC weights...")
        pc_joint.randomize_weights()
        pc_marginal.set_weights(pc_joint.get_weights())

    logger.log_trace("Total PC nodes: " + str(len(pc_joint.nodes)) + ".")
    logger.log_trace("Total PC sum nodes: " + str(len(pc_joint.sum_nodes)) + ".")
    logger.log_trace("Total PC product nodes: " + str(len(pc_joint.product_nodes)) + ".")
    logger.log_trace("Total PC leaf nodes: " + str(len(pc_joint.leaf_nodes)) + ".")
    logger.log_trace("PC depth: " + str(pc_joint.depth) + ".")
    logger.log_trace("PC leaf node setting dimension: (" + str(int(pc_settings_joint.shape[0])) + ", " + str(int(pc_settings_joint.shape[1])) + ").")

    progress_bar = tqdm.tqdm(total = header.config_neural["epochs"], position = 0)

    progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_neural["epochs"]:
        progress_bar.n = epoch
        progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_neural, pc_joint, pc_marginal, data_loader_train, optimizer_neural, optimizer_pc, device, batch_step_train)
        (accuracy_classification_validation_epoch, loss_validation_epoch, batch_step_validate) = validate(model_neural, pc_joint, pc_marginal, data_loader_validation, device, batch_step_validate)

        learning_rate_scheduler.step(loss_validation_epoch)
        learning_rate_scheduler_pc.step(loss_validation_epoch)

        logger.log_info("Validation classification accuracy: " + str(accuracy_classification_validation_epoch) + ".")

        if accuracy_classification_validation_epoch > accuracy_classification_validation_best or epoch == 1:
            accuracy_classification_validation_best = accuracy_classification_validation_epoch
            wandb.log({"validation/epoch/accuracy_classification_best": accuracy_classification_validation_best})
            utility.saveCheckpoint(header.config_neural["file_name_checkpoint_best"], model_neural)
            utility.saveCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_joint, True)

        utility.saveCheckpoint(header.config_neural["file_name_checkpoint"], model_neural)
        utility.saveCheckpoint(header.config_pc["file_name_checkpoint"], pc_joint, True)

        epoch += 1

    progress_bar.close()

    logger.log_info("Best validation classification accuracy: " + str(accuracy_classification_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_classification_best"] = accuracy_classification_validation_best

    if header.npc_pc_backward:
        settings_marginal = torch.full((1, pc_settings_joint.shape[1]), -1).to(device)
        logger.log_info("Normalizing PC weights...")

        utility.loadCheckpoint(header.config_pc["file_name_checkpoint"], pc_marginal, True)
        pc_marginal(settings_marginal)
        pc_marginal.normalize_weights(header.config_pc["epsilon_smoothing"])
        utility.saveCheckpoint(header.config_pc["file_name_checkpoint"], pc_marginal, True)

        utility.loadCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_marginal, True)
        pc_marginal(settings_marginal)
        pc_marginal.normalize_weights(header.config_pc["epsilon_smoothing"])
        utility.saveCheckpoint(header.config_pc["file_name_checkpoint_best"], pc_marginal, True)

        pc_marginal.set_leaf_nodes_categorical(pc_settings_marginal)

    wandb.log({"testing/epoch/step": batch_step_test})
    test_npc.test(model_neural, pc_joint, pc_marginal, pc_settings_joint, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
