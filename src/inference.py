#!/usr/bin/env python3

import dataset
import composition
import header
import logger
import network
import sklearn.metrics
import torch
import torch.nn
import tqdm
import utility
import visualize

def main():
    utility.processArgumentsInference()

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    accuracy_epoch_baseline = 0
    accuracy_epoch_composed = 0
    ground_truths_epoch_baseline = []
    output_list_baseline = []
    output_list_composed = []
    predictions_epoch_baseline = []
    predictions_epoch_composed = []
    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_baseline["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_baseline = network.createModelBaseline(device)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)
    model_decomposed = network.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    spn_matrix_a = torch.load(header.file_path_spn_matrix_a).float()
    spn_matrix_a = spn_matrix_a.to(device)

    utility.loadCheckpointBest(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint_best"], model_baseline)
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)

    progress_bar = tqdm.tqdm(total = len(data_loader_test), position = 0, leave = False)

    model_baseline.eval()
    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Inference progress")

    with torch.no_grad():
        for (batch_index, (input, _, labels)) in enumerate(data_loader_test):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            with torch.set_grad_enabled(False):
                output_baseline = model_baseline(input)
                (outputs_decomposed, _) = model_decomposed(input)

                output_baseline = utility.applySoftmax(output_baseline)
                outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed)

                output_composed = composition.Composition.spn(outputs_decomposed, spn_matrix_a, device)

                (_, predictions_baseline) = torch.max(output_baseline, 1)
                (_, predictions_composed) = torch.max(output_composed, 1)

                corrects_baseline = torch.sum(predictions_baseline == labels.data).item()
                corrects_composed = torch.sum(predictions_composed == labels.data).item()

            accuracy_epoch_baseline += corrects_baseline
            accuracy_epoch_composed += corrects_composed

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            ground_truths_epoch_baseline += labels.data.tolist()
            predictions_epoch_baseline += predictions_baseline.tolist()
            predictions_epoch_composed += predictions_composed.tolist()

            if header.visualize_show or header.visualize_save:
                visualize.visualize(input, composition, dataset_test, labels, output_baseline, output_composed)

    progress_bar.close()

    accuracy_epoch_baseline /= len(data_loader_test.dataset)
    accuracy_epoch_composed /= len(data_loader_test.dataset)

    precision_epoch_baseline = sklearn.metrics.precision_score(ground_truths_epoch_baseline, predictions_epoch_baseline, average = "macro", zero_division = 0)
    precision_epoch_composed = sklearn.metrics.precision_score(ground_truths_epoch_baseline, predictions_epoch_composed, average = "macro", zero_division = 0)
    recall_epoch_baseline = sklearn.metrics.recall_score(ground_truths_epoch_baseline, predictions_epoch_baseline, average = "macro", zero_division = 0)
    recall_epoch_composed = sklearn.metrics.recall_score(ground_truths_epoch_baseline, predictions_epoch_composed, average = "macro", zero_division = 0)
    output_list_baseline += [accuracy_epoch_baseline, precision_epoch_baseline, recall_epoch_baseline]
    output_list_composed += [accuracy_epoch_composed, precision_epoch_composed, recall_epoch_composed]

    logger.log_info("Baseline inference accuracy: " + str(accuracy_epoch_baseline) + ".")
    logger.log_info("Composed inference accuracy: " + str(accuracy_epoch_composed) + ".")

    utility.logTestOutput(header.config_baseline, output_list_baseline)
    utility.logTestOutput(header.config_baseline, output_list_composed, True)

    return

if __name__ == "__main__":
    main()
