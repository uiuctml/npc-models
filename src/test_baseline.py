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

def test(model_baseline, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint_best"], model_baseline)

    accuracy_task_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    model_baseline.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, _, labels, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            output = model_baseline(input)
            (_, predictions) = torch.max(output, 1)

            corrects = torch.sum(predictions == labels).item()

            accuracy_task_batch = corrects / input.size(0)
            accuracy_task_epoch += corrects

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy_task": accuracy_task_batch})
            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    accuracy_task_epoch /= len(data_loader.dataset)

    wandb.log({"testing/epoch/accuracy_task": accuracy_task_epoch})
    wandb.summary["testing/epoch/accuracy_task"] = accuracy_task_epoch

    logger.log_info("Testing task accuracy: " + str(accuracy_task_epoch) + ".")

    return batch_step

def main():
    argument.processArgumentsTestBaseline()

    utility.setSeed(header.config_baseline["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_baseline, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_baseline)
    dataset_test = dataset.NPCDataset(header.config_baseline["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_baseline["data_loader_batch_size"], shuffle = False, num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_baseline = model.createModelBaseline(device)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)

    test(model_baseline, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
