#!/usr/bin/env python3

import composition as comp
import dataset as dset
import header
import logger
import network
import sys
import torch
import torch.nn
import torchvision
import tqdm
import utility
import visualize

def main():
    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    run_name_baseline = ""
    run_name_decomposed = ""

    if len(sys.argv) > 2:
        run_name_baseline = sys.argv[1]
        run_name_decomposed = sys.argv[2]
    else:
        logger.log_error("Run names missing. Quit.")
        return

    if not utility.initializeRunNameBaseline(run_name_baseline) or header.run_name_baseline == "":
        logger.log_error("Baseline run name missing. Quit.")
        return

    if not utility.initializeRunNameDecomposed(run_name_decomposed) or header.run_name_decomposed == "":
        logger.log_error("Decomposed run name missing. Quit.")
        return

    accuracy_epoch_baseline = 0
    accuracy_epoch_composed = 0
    accuracy_epoch_decomposed_list = []
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((header.config_decomposed["model_input_height"], header.config_decomposed["model_input_width"])),
        torchvision.transforms.ToTensor(),
    ])
    dataset_original = torchvision.datasets.ImageFolder(header.config_baseline["dir_dataset_test"], dataset_transforms)
    dataset_test = dset.DatasetDecomposed(header.config_decomposed["dir_dataset_test"], dataset_original.classes, dataset_transforms)
    config_dataset = dataset_test.config
    class_count_original = len(dataset_original.classes)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    composition = comp.Composition(dataset_test, device)
    model_baseline = network.createModelBaseline(class_count_original)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)
    model_decomposed = network.createModelDecomposed(config_dataset)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)

    for _ in config_dataset["datasets"]:
        accuracy_epoch_decomposed_list.append(0)

    utility.loadCheckpointBest(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint_best"], model_baseline)
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)

    progress_bar = tqdm.tqdm(total = len(data_loader_test), position = 0, leave = False)

    model_baseline.eval()
    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Inference progress")

    with torch.no_grad():
        for (batch_index, (input, labels_decomposed, labels_original)) in enumerate(data_loader_test):
            input = input.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output_baseline = model_baseline(input)
                outputs_decomposed = model_decomposed(input)

                output_baseline = composition.applySoftmax(output_baseline)
                outputs_decomposed = composition.applySoftmaxDecomposed(outputs_decomposed)

                output_composed = composition.compose(outputs_decomposed)

                (_, predictions_baseline) = torch.max(output_baseline, 1)
                (_, predictions_composed) = torch.max(output_composed, 1)

                corrects_baseline = torch.sum(predictions_baseline == labels_original.data).item()
                corrects_composed = torch.sum(predictions_composed == labels_original.data).item()

                for i in range(0, len(config_dataset["datasets"])):
                    (_, predictions_decomposed) = torch.max(outputs_decomposed[i], 1)

                    corrects_decomposed = torch.sum(predictions_decomposed == labels_decomposed[:, i].data).item()
                    accuracy_epoch_decomposed_list[i] += corrects_decomposed

            accuracy_epoch_baseline += corrects_baseline
            accuracy_epoch_composed += corrects_composed

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            if header.visualize_show or header.visualize_save:
                visualize.visualize(input, composition, dataset_test, labels_decomposed, labels_original, output_baseline, outputs_decomposed, output_composed)

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["datasets"]):
        accuracy_epoch_decomposed_list[i] /= len(data_loader_test.dataset)
        logger.log_info("Decomposed testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_epoch_decomposed_list[i]) + ".")

    accuracy_epoch_baseline /= len(data_loader_test.dataset)
    accuracy_epoch_composed /= len(data_loader_test.dataset)

    logger.log_info("Composed inference accuracy: " + str(accuracy_epoch_composed) + ".")
    logger.log_info("Baseline inference accuracy: " + str(accuracy_epoch_baseline) + ".")

    return

if __name__ == "__main__":
    main()
