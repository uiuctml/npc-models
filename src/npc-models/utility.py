import datetime
import header
import logger
import natsort
import numpy
import os
import random
import socket
import torch
import torchvision
import wandb

def applySoftmax(output):
    return torch.nn.functional.softmax(output, dim = 1)

def applySoftmaxDecomposed(outputs_decomposed):
    outputs_decomposed_softmax = []

    for i in range(len(outputs_decomposed)):
        outputs_decomposed_softmax.append(applySoftmax(outputs_decomposed[i]))

    return outputs_decomposed_softmax

def computeAccuracyDecomposed(outputs, labels, device):
    count_attributes = len(outputs)
    corrects = []

    for i in range(count_attributes):
        masks_labels = (labels[i] > 0)
        counts_values = torch.sum(masks_labels, dim = 1)
        masks_predictions = []

        for (output_batch, count_value_batch) in zip(outputs[i], counts_values):
            mask_prediction_batch = torch.zeros(output_batch.shape, dtype = torch.bool)
            (_, prediction_batch) = torch.topk(output_batch, count_value_batch)
            mask_prediction_batch[prediction_batch] = True
            masks_predictions.append(mask_prediction_batch)

        masks_predictions = torch.stack(masks_predictions, dim = 0).to(device)
        corrects_attribute = torch.all(masks_predictions == masks_labels, dim = 1)

        corrects.append(corrects_attribute)

    corrects = torch.stack(corrects, dim = 1)
    corrects = torch.all(corrects, dim = 1)
    accuracy = (torch.sum(corrects) / corrects.size(0)).item()

    return accuracy

def createTransform(config):
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config["model_input_height"], config["model_input_width"])),
        torchvision.transforms.ToTensor()
    ])

    return dataset_transforms

def defineMetrics():
    wandb.define_metric("testing/batch/step")
    wandb.define_metric("testing/epoch/step")
    wandb.define_metric("training/batch/step")
    wandb.define_metric("training/epoch/step")
    wandb.define_metric("validation/batch/step")
    wandb.define_metric("validation/epoch/step")

    wandb.define_metric("testing/batch/*", step_metric = "testing/batch/step")
    wandb.define_metric("testing/epoch/*", step_metric = "testing/epoch/step")
    wandb.define_metric("training/batch/*", step_metric = "training/batch/step")
    wandb.define_metric("training/epoch/*", step_metric = "training/epoch/step")
    wandb.define_metric("validation/batch/*", step_metric = "validation/batch/step")
    wandb.define_metric("validation/epoch/*", step_metric = "validation/epoch/step")

    return

def generateRunName(seed, type, method):
    date_time_list = list(datetime.datetime.now().timetuple())[:-4]
    run_name = str(seed)

    run_name += "."
    run_name += header.dataset_prefix
    run_name += "."
    run_name += type
    run_name += "."
    run_name += method

    for entry in date_time_list:
        run_name += "."
        run_name += str(entry)

    run_name += "."
    run_name += socket.gethostname()

    return run_name

def getBinaryLabelsDecomposed(labels_decomposed):
    labels_decomposed_binary = []

    for label_decomposed in labels_decomposed:
        labels_decomposed_binary.append((label_decomposed > 0).float())

    return torch.cat(labels_decomposed_binary, dim = 1)

def getBinaryLabelsClass(labels_class, data_loader):
    return torch.nn.functional.one_hot(labels_class, num_classes = len(data_loader.dataset.labels_class)).float()

def getLabelsAttribute(dataset_config):
    labels_attribute = {}

    for attribute in dataset_config["attributes"]:
        if "" in attribute["labels"]:
            attribute["labels"].remove("")

        labels_attribute[attribute["name"]] = attribute["labels"]

    return labels_attribute

def getLabelsClass(dataset_config):
    if "instance_wise" in dataset_config and dataset_config["instance_wise"]:
        labels_class = []
        labels_class_set = set()

        for image_name in dataset_config["mappings"].keys():
            class_name = image_name.split('/')[0]

            if class_name not in labels_class_set:
                labels_class.append(class_name)
                labels_class_set.add(class_name)

        return natsort.natsorted(labels_class)
    else:
        return list(dataset_config["mappings"].keys())

def getIndicesFromLabelsAttribute(labels_attribute):
    indices = {}

    for attribute in labels_attribute.keys():
        labels_to_indices = {}

        for i in range(len(labels_attribute[attribute])):
            labels_to_indices[labels_attribute[attribute][i]] = i

        indices[attribute] = labels_to_indices

    return indices

def getIndicesFromLabelsClass(labels_class):
    labels_to_indices = {}

    for i in range(len(labels_class)):
        labels_to_indices[labels_class[i]] = i

    return labels_to_indices

def generatePCSettings(config_dataset, device):
    attribute_ranges = []
    labels_attribute = getLabelsAttribute(config_dataset)
    labels_class = getLabelsClass(config_dataset)

    for attribute in labels_attribute.keys():
        attribute_count = len(labels_attribute[attribute])
        attribute_range = torch.Tensor(range(attribute_count))
        attribute_range = attribute_range.to(device)
        attribute_ranges.append(attribute_range)

        logger.log_trace("Number of attribute labels for \"" + attribute + "\": " + str(attribute_count) + ".")

    class_count = len(labels_class)
    class_range = torch.Tensor(range(class_count))
    class_range = class_range.to(device)

    logger.log_trace("Number of class labels: " + str(class_count) + ".")

    pc_settings = torch.cartesian_prod(*attribute_ranges)

    class_range = class_range.repeat_interleave(pc_settings.shape[0]).reshape(-1, 1)
    pc_settings = pc_settings.repeat(class_count, 1)
    pc_settings = torch.cat((pc_settings, class_range), 1)

    return pc_settings

def loadCheckpoint(file_name_checkpoint, model, pc = False):
    if not os.path.isdir(header.checkpoint_dir):
        logger.log_fatal("Checkpoint directory \"" + header.checkpoint_dir + "\" missing.")
        exit(-1)

    file_path_checkpoint = os.path.join(header.checkpoint_dir, file_name_checkpoint)

    if os.path.isfile(file_path_checkpoint):
        checkpoint = torch.load(file_path_checkpoint)

        if not pc:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.set_weights(checkpoint["weights"])

        logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
    else:
        logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
        exit(-1)

    return

def lossNegativeLogLikelihood(output, label):
    label = label.reshape(-1, 1)
    output = torch.gather(output, 1, label)
    negative_log_likelihood = -1 * torch.log(output)

    return negative_log_likelihood.mean()

def saveCheckpoint(file_name_checkpoint, model, pc = False):
    if not os.path.isdir(header.checkpoint_dir):
        os.makedirs(header.checkpoint_dir, exist_ok = True)

    checkpoint = None

    if not pc:
        checkpoint = {"model_state_dict": model.state_dict()}
    else:
        checkpoint = {"weights": model.get_weights()}

    file_path_checkpoint = os.path.join(header.checkpoint_dir, file_name_checkpoint)

    torch.save(checkpoint, file_path_checkpoint)

    try:
        wandb.save(file_path_checkpoint, base_path = header.checkpoint_dir)
    except:
        pass
    else:
        logger.log_trace("Saved checkpoint \"" + file_name_checkpoint + "\" to Weights & Biases.")

    logger.log_trace("Saved checkpoint \"" + file_name_checkpoint + "\".")

    return

def setSeed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    return
