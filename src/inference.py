#!/usr/bin/env python3

import config
import dataset as dset
import decision as deci
import logger
import network
import sys
import torch
import torch.nn
import tqdm
import utility
import wandb

def main():
    if len(sys.argv) > 2:
        if sys.argv[1].split(".")[0] != config.run_name_baseline_keyword:
            logger.log_error("Invalid baseline run name. Quit.")
            return

        if sys.argv[2].split(".")[0] != config.run_name_decomposed_keyword:
            logger.log_error("Invalid decomposed run name. Quit.")
            return

        config.run_name_baseline = sys.argv[1]
        config.run_name_decomposed = sys.argv[2]

        config.config_baseline["file_name_checkpoint"] = config.run_name_baseline + ".tar"
        config.config_baseline["file_name_checkpoint_best"] = config.run_name_baseline + ".best.tar"
        config.config_decomposed["file_name_checkpoint"] = config.run_name_decomposed + ".tar"
        config.config_decomposed["file_name_checkpoint_best"] = config.run_name_decomposed + ".best.tar"
    else:
        logger.log_error("Run names missing. Quit.")
        return

    wandb.init(config = config.config_baseline, mode = "disabled")

    wandb.finish()
    wandb.init(config = config.config_decomposed, mode = "disabled")

    accuracy_epoch_decomposed = 0
    dataset_decomposed = dset.DatasetGenerated(root = wandb.config.dir_dataset)
    device = torch.device("cuda")
    decision = deci.Decision(dataset_decomposed, device)
    model_decomposed = network.DecomposedNetworkA(dataset_decomposed)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)

    data_loader = utility.loadCheckpointBest(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, model_decomposed)

    if data_loader is None:
        logger.log_error("Data loader missing.")
        return

    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    wandb.finish()

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Inference progress")

    with torch.no_grad():
        for (batch_index, (input, labels)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                outputs_decomposed = model_decomposed(input)
                (output_decomposed, labels) = decision.make(outputs_decomposed, labels, input.size(0))
                (_, predictions_decomposed) = torch.max(output_decomposed, 1)

                corrects_decomposed = torch.sum(predictions_decomposed == labels.data).item()

            accuracy_batch_decomposed = corrects_decomposed / input.size(0)
            accuracy_epoch_decomposed += corrects_decomposed

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

    progress_bar.close()

    accuracy_epoch_decomposed /= len(data_loader.dataset)

    logger.log_info("Testing accuracy for decomposed network: " + str(accuracy_epoch_decomposed) + ".")

    return

if __name__ == "__main__":
    main()
