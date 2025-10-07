#!/usr/bin/env python3

import argparse
import dataset
import header
import logger
import model
import torch
import tqdm
import utility
import wandb

def initializeRunName(run_name = ""):
    if run_name != "":
        if run_name.split(".")[2] != header.config_blackbox["type"]:
            logger.log_fatal("Not a blackbox run name. Quit.")
            exit(-1)
        elif run_name.split(".")[1] != header.dataset_prefix:
            logger.log_fatal("Blackbox run name dataset is not \"" + header.dataset_prefix + "\". Quit.")
            exit(-1)
        elif run_name.split(".")[3] != "resnet34":
            logger.log_fatal("Unknown blackbox model \"" + run_name.split(".")[3] + "\". Quit.")
            exit(-1)

        header.config_blackbox["run_name"] = run_name
    else:
        header.config_blackbox["run_name"] = utility.generateRunName(header.config_blackbox["seed"], header.config_blackbox["type"], "resnet34")

    if header.config_blackbox["run_name"] == "":
        logger.log_fatal("Missing blackbox run name. Quit.")
        exit(-1)

    header.config_blackbox["file_name_checkpoint"] = header.config_blackbox["run_name"] + header.checkpoint_postfix
    header.config_blackbox["file_name_checkpoint_best"] = header.config_blackbox["run_name"] + header.checkpoint_postfix_best

    return

def processArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Seed.")
    arguments = parser.parse_args()

    initializeRunName(arguments.run_name)

    if arguments.seed is not None:
        header.config_blackbox["seed"] = arguments.seed

    logger.log_trace("Run name: \"" + header.config_blackbox["run_name"] + "\".")
    logger.log_trace("Seed: " + str(header.config_blackbox["seed"]) + ".")

    return

def test(model_blackbox, data_loader, device, batch_step):
    utility.loadCheckpoint(header.config_blackbox["file_name_checkpoint_best"], model_blackbox)

    accuracy_classification_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    model_blackbox.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, _, labels, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            output = model_blackbox(input)
            (_, predictions) = torch.max(output, 1)

            corrects = torch.sum(predictions == labels).item()

            accuracy_classification_batch = corrects / input.size(0)
            accuracy_classification_epoch += corrects

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy_classification": accuracy_classification_batch})
            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    accuracy_classification_epoch /= len(data_loader.dataset)

    wandb.log({"testing/epoch/accuracy_classification": accuracy_classification_epoch})
    wandb.summary["testing/epoch/accuracy_classification"] = accuracy_classification_epoch

    logger.log_info("Testing classification accuracy: " + str(accuracy_classification_epoch) + ".")

    return batch_step

def main():
    processArguments()

    utility.setSeed(header.config_blackbox["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_blackbox, mode = "disabled")

    dataset_transforms = utility.createTransforms(header.config_blackbox)
    dataset_test = dataset.NPCDataset(header.config_blackbox["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_blackbox["batch_size"], shuffle = False, num_workers = header.config_blackbox["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_blackbox = model.ResNet34(dataset_test.config, device)
    model_blackbox = torch.nn.DataParallel(model_blackbox)
    model_blackbox = model_blackbox.to(device)

    test(model_blackbox, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
