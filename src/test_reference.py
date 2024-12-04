#!/usr/bin/env python3

import argument
import dataset
import header
import logger
import model
import sklearn.metrics
import torch
import tqdm
import type
import utility
import wandb

def test(model_reference, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_reference["dir_checkpoints"], header.config_reference["file_name_checkpoint_best"], model_reference)

    accuracy_attribute_epoch = 0
    accuracy_task_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    threshold_accuracy_attribute = 0.5
    threshold_accuracy_task = 0.5

    if header.config_reference["model"] == type.ModelReference.cbm.name:
        threshold_accuracy_attribute = 0
        threshold_accuracy_task = 0
    elif header.config_reference["model"] == type.ModelReference.cem.name:
        threshold_accuracy_task = 0

    model_reference.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_decomposed = utility.getBinaryLabelsDecomposed(labels_decomposed, device)
            labels_original = utility.getBinaryLabelsOriginal(labels_original, data_loader, device)

            (output_neck, output_head) = model_reference(input)

            accuracy_attribute_batch = sklearn.metrics.accuracy_score(labels_decomposed.cpu(), (output_neck > threshold_accuracy_attribute).cpu())
            accuracy_task_batch = sklearn.metrics.accuracy_score(labels_original.cpu(), (output_head > threshold_accuracy_task).cpu())

            accuracy_attribute_epoch += accuracy_attribute_batch
            accuracy_task_epoch += accuracy_task_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/attribute/accuracy": accuracy_attribute_batch})
            wandb.log({"testing/batch/accuracy": accuracy_task_batch})
            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    accuracy_attribute_epoch /= len(data_loader)
    accuracy_task_epoch /= len(data_loader)

    wandb.log({"testing/epoch/attribute/accuracy": accuracy_attribute_epoch})
    wandb.log({"testing/epoch/accuracy": accuracy_task_epoch})

    wandb.summary["testing/epoch/attribute/accuracy"] = accuracy_attribute_epoch
    wandb.summary["testing/epoch/accuracy"] = accuracy_task_epoch

    logger.log_info("Testing attribute accuracy: " + str(accuracy_attribute_epoch) + ".")
    logger.log_info("Testing accuracy: " + str(accuracy_task_epoch) + ".")

    return batch_step

def main():
    argument.processArgumentsTestReference()

    utility.setSeed(header.config_reference["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_reference, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_reference)
    dataset_test = dataset.VISATDataset(header.config_reference["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_reference["data_loader_batch_size"], shuffle = False, num_workers = header.config_reference["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_reference = model.createModelReference(device)
    model_reference = torch.nn.DataParallel(model_reference)
    model_reference = model_reference.to(device)

    test(model_reference, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
