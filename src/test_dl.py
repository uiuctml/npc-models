#!/usr/bin/env python3

import argument
import dataset
import header
import logger
import model
import torch
import tqdm
import utility
import wandb

def test(model_decomposed, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)

    accuracy_attribute_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    tv_distance_epoch = 0
    tv_distances_epoch = []

    for _ in data_loader.dataset.config["attributes"]:
        tv_distances_epoch.append(0)

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels, _, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)

            for i in range(len(labels)):
                labels[i] = labels[i].to(device, non_blocking = True)

            (output, _) = model_decomposed(input)

            output = utility.applySoftmaxDecomposed(output)
            accuracy_attribute_batch = utility.computeAccuracyDecomposed(output, labels, device)

            accuracy_attribute_epoch += accuracy_attribute_batch

            for i in range(len(data_loader.dataset.config["attributes"])):
                tv_distance_batch = 0.5 * torch.sum(torch.abs(output[i] - labels[i]), dim = 1)
                tv_distances_epoch[i] += torch.sum(tv_distance_batch).item()

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy_attribute": accuracy_attribute_batch})
            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)

    for i in range(len(data_loader.dataset.config["attributes"])):
        tv_distances_epoch[i] /= len(data_loader.dataset)

    tv_distance_epoch = sum(tv_distances_epoch) / len(tv_distances_epoch)

    wandb.log({"testing/epoch/accuracy_attribute": accuracy_attribute_epoch})
    wandb.log({"testing/epoch/tv_distance_attribute": tv_distance_epoch})

    wandb.summary["testing/epoch/accuracy_attribute"] = accuracy_attribute_epoch
    wandb.summary["testing/epoch/tv_distance_attribute"] = tv_distance_epoch

    logger.log_info("Testing attribute TV distance: " + str(tv_distance_epoch) + ".")
    logger.log_info("Testing attribute accuracy: " + str(accuracy_attribute_epoch) + ".")

    return batch_step

def main():
    argument.processArgumentsTestDecomposed()

    utility.setSeed(header.config_decomposed["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_decomposed, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.NPCDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)

    test(model_decomposed, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
