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
    if len(sys.argv) > 1:
        config.run_name_decomposed = sys.argv[1]
        config.config_decomposed["file_name_checkpoint"] = config.run_name_decomposed + ".tar"
        config.config_decomposed["file_name_checkpoint_best"] = config.run_name_decomposed + ".best.tar"
    else:
        logger.log_error("Run name missing. Quit.")
        return

    wandb.init(config = config.config_decomposed, mode = "disabled")

    accuracy_epoch = 0
    dataset = dset.DatasetGenerated(root = wandb.config.dir_dataset)
    device = torch.device("cuda")
    decision = deci.Decision(dataset, device)
    model = network.DecomposedNetworkA(dataset)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    data_loader = utility.loadCheckpointBest(wandb.config.dir_checkpoints, wandb.config.file_name_checkpoint_best, model)

    if data_loader is None:
        logger.log_error("Data loader missing.")
        return

    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size = wandb.config.data_loader_batch_size, shuffle = wandb.config.data_loader_shuffle, num_workers = wandb.config.data_loader_worker_count, pin_memory = True)
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    model.eval()
    progress_bar.set_description_str("[INFO]: Inference progress")

    with torch.no_grad():
        for (batch_index, (input, labels)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                outputs = model(input)
                (output, labels) = decision.make(outputs, labels, input.size(0))
                (_, predictions) = torch.max(output, 1)
                
                corrects = torch.sum(predictions == labels.data).item()

            accuracy_batch = corrects / input.size(0)
            accuracy_epoch += corrects

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

    progress_bar.close()

    accuracy_epoch /= len(data_loader.dataset)

    logger.log_info("Testing accuracy: " + str(accuracy_epoch) + ".")

    return

if __name__ == "__main__":
    main()
