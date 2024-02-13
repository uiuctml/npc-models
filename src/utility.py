import argparse
import cv2
import datetime
import header
import itertools
import logger
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
    for task_index in range(0, len(outputs_decomposed)):
        outputs_decomposed[task_index] = torch.nn.functional.softmax(outputs_decomposed[task_index], dim = 1)

    return outputs_decomposed

def computeCovarianceRegularization(features):
    loss = 0
    features_centered = []

    # Mean-center all attribute feature tensors across rows
    for feature in features:
        means_col = torch.mean(feature, axis = 0)
        features_centered.append(feature.detach().clone() - means_col)

    # Obtain unique pairs of attribute feature tensors
    feature_pairs = list(itertools.combinations(features_centered, 2))

    for feature_pair in feature_pairs:
        # Compute pair-wise covariance matrix
        convariance = torch.matmul(feature_pair[0].t(), feature_pair[1])

        # Add Frobenius norm to total loss
        loss += torch.sum(torch.square(convariance))

    # Multiply total loss with tunable regularization factor
    loss *= header.config_decomposed["factor_loss_covariance"]

    return loss

def computeL2Norm(parameters):
    parameters_list = []

    for parameter in parameters:
        parameters_list.append(parameter.view(-1))

    return torch.square(torch.cat(parameters_list)).sum().item()

def createTransform(config):
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config["model_input_height"], config["model_input_width"])),
        torchvision.transforms.ToTensor()
    ])

    if config["input_grayscale"]:
        dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config["model_input_height"], config["model_input_width"])),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])

    return dataset_transforms

def getLabelsAttribute(dataset_config):
    labels_attribute = {}

    for attribute in dataset_config["attributes"]:
        attribute["labels"].remove("")
        labels_attribute[attribute["name"]] = attribute["labels"]

    return labels_attribute

def getLabelsOriginal(dataset_config):
    return list(dataset_config["mappings"].keys())

def getIndicesFromLabelsAttribute(labels_attribute):
    indices = {}

    for attribute in labels_attribute.keys():
        labels_to_indices = {}

        for i in range(len(labels_attribute[attribute])):
            labels_to_indices[labels_attribute[attribute][i]] = i

        indices[attribute] = labels_to_indices

    return indices

def getIndicesFromLabelsOriginal(labels_original):
    labels_to_indices = {}

    for i in range(len(labels_original)):
        labels_to_indices[labels_original[i]] = i

    return labels_to_indices

def wAndBGenerateRunName(model_name, model_type, seed, fine_tuning):
    date_time_list = list(datetime.datetime.now().timetuple())[:-4]
    run_name = model_name
    run_name += "."
    run_name += model_type
    run_name += "."

    if fine_tuning:
        run_name += "ft"
    else:
        run_name += "ftl"

    run_name += "."
    run_name += str(seed)

    for entry in date_time_list:
        run_name += "."
        run_name += str(entry)

    run_name += "."
    run_name += socket.gethostname()

    return run_name

def initializeArgumentsAttack():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "WandB run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")
    parser.add_argument("-d", "--test-dataset-dir", type = str, default = "", help = "Directory of dataset testing split.")
    parser.add_argument("-a", "--attribute", type = int, default = 0, help = "Targeted attribute.")

    return parser.parse_args()

def initializeArgumentsTest():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "WandB run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")
    parser.add_argument("-d", "--test-dataset-dir", type = str, default = "", help = "Directory of dataset testing split.")

    return parser.parse_args()

def initializeArgumentsTrain():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "WandB run name.")
    parser.add_argument("-m", "--model", type = str, default = "", help = "Model to train.")
    parser.add_argument("-b", "--batch-size", type = int, default = 512, help = "Batch size.")
    parser.add_argument("-e", "--epochs", type = int, default = 100, help = "Number of training epochs.")
    parser.add_argument("-f", "--fine-tune", type = int, default = 1, help = "Whether to perform backbone fine-tuning.")
    parser.add_argument("-c", "--use-covariance-loss", type = int, default = 0, help = "Whether to use covariance loss.")
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")

    return parser.parse_args()

def initializeRunNameBaseline(run_name):
    resume = False
    if run_name != "":
        if run_name.split(".")[0] != header.run_name_baseline_keyword:
            logger.log_fatal("Invalid baseline run name. Quit.")
            exit(-1)

        header.run_name_baseline = run_name
        resume = True
    else:
        header.run_name_baseline = wAndBGenerateRunName(header.run_name_baseline_keyword, header.config_baseline["model"], header.config_baseline["seed"], header.config_baseline["fine_tuning"])

    header.config_baseline["file_name_checkpoint"] = header.run_name_baseline + ".tar"
    header.config_baseline["file_name_checkpoint_best"] = header.run_name_baseline + ".best.tar"
    header.config_baseline["run_name"] = header.run_name_baseline

    return resume

def initializeRunNameDecomposed(run_name):
    resume = False
    if run_name != "":
        if run_name.split(".")[0] != header.run_name_decomposed_keyword:
            logger.log_fatal("Invalid decomposed run name. Quit.")
            exit(-1)

        header.run_name_decomposed = run_name
        resume = True
    else:
        header.run_name_decomposed = wAndBGenerateRunName(header.run_name_decomposed_keyword, header.config_decomposed["model"], header.config_decomposed["seed"], header.config_decomposed["fine_tuning"])

    header.config_decomposed["file_name_checkpoint"] = header.run_name_decomposed + ".tar"
    header.config_decomposed["file_name_checkpoint_best"] = header.run_name_decomposed + ".best.tar"
    header.config_decomposed["run_name"] = header.run_name_decomposed

    return resume

def loadCheckpoint(dir_checkpoints, file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_schedulers, model, optimizers):
    if wandb.run.resumed:
        if not os.path.isdir(dir_checkpoints):
            os.makedirs(dir_checkpoints, exist_ok = True)

        try:
            wandb.restore(file_name_checkpoint, root = dir_checkpoints)
        except:
            pass
        else:
            logger.log_info("Restored checkpoint \"" + file_name_checkpoint + "\" from Weights & Biases.")

        file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

        if os.path.isfile(file_path_checkpoint):
            checkpoint = torch.load(file_path_checkpoint)
            accuracy_validation_best = checkpoint["accuracy_validation_best"]
            batch_step_train = checkpoint["batch_step_train"]
            batch_step_validate = checkpoint["batch_step_validate"]
            criterions = checkpoint["criterions"]
            epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])

            for (learning_rate_scheduler, learning_rate_scheduler_state_dict) in zip(learning_rate_schedulers, checkpoint["learning_rate_scheduler_state_dict_list"]):
                learning_rate_scheduler.load_state_dict(learning_rate_scheduler_state_dict)

            for (optimizer, optimizer_state_dict) in zip(optimizers, checkpoint["optimizer_state_dict_list"]):
                optimizer.load_state_dict(optimizer_state_dict)

            logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
        else:
            logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
            exit(-1)

    return (accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch)

def loadCheckpointBest(dir_checkpoints, file_name_checkpoint, model):
    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok = True)

    try:
        wandb.restore(file_name_checkpoint, root = dir_checkpoints)
    except:
        pass
    else:
        logger.log_info("Restored checkpoint \"" + file_name_checkpoint + "\" from Weights & Biases.")

    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    if os.path.isfile(file_path_checkpoint):
        checkpoint = torch.load(file_path_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
    else:
        logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
        exit(-1)

    return

def logTestOutput(output, config, config_dataset, composed = False):
    if config["log_test_output"] == False:
        return

    dir_dataset_test = config["dir_dataset_test"]
    dir_test_output = config["dir_test_output"]
    file_empty = False
    file_line = ""
    file_path_test_output = ""

    if dir_dataset_test[-1] == '/':
        dir_dataset_test = dir_dataset_test[:-1]

    if not os.path.isdir(dir_test_output):
        os.makedirs(dir_test_output, exist_ok = True)

    if dir_dataset_test.split('/')[-1] == "test":
        file_path_test_output = os.path.join(dir_test_output, config["file_name_test_output_clean"])
    else:
        file_path_test_output = os.path.join(dir_test_output, config["file_name_test_output_attacked"])

    if not os.path.isfile(file_path_test_output):
        file_empty = True
    elif os.stat(file_path_test_output).st_size == 0:
        file_empty = True

    file_line += str(config["seed"])
    file_line += "\t"
    file_line += config["run_name"]
    file_line += "\t"
    file_line += dir_dataset_test.split('/')[-1]
    file_line += "\t"
    file_line += str(composed)

    for value in output:
        file_line += "\t"
        file_line += str(value)

    file_line += "\n"

    with open(file_path_test_output, "a") as file_test_output:
        if file_empty:
            if config["type"] == "baseline":
                file_test_output.write("# seed\tmodel\tattack\tcomposed\taccuracy\tprecision\trecall\n")
            elif config["type"] == "decomposed":
                file_line_header = "# seed\tmodel\tattack\tcomposed"

                for data_type in ["accuracy", "precision", "recall"]:
                    for attribute in config_dataset["attributes"]:
                        attribute_name = attribute["name"]
                        file_line_header += "\t"
                        file_line_header += data_type + "_" + attribute_name

                file_line_header += "\n"
                file_test_output.write(file_line_header)
            else:
                logger.log_fatal("Unknown configuration type")
                exit(-1)

        file_test_output.write(file_line)

    logger.log_info("Logged test output to \"" + file_path_test_output + "\".")

    return

def processArgumentsAttack():
    arguments = initializeArgumentsAttack()

    header.config_decomposed["seed"] = arguments.seed

    if arguments.test_dataset_dir != "":
        header.config_decomposed["dir_dataset_test"] = arguments.test_dataset_dir

    if not initializeRunNameDecomposed(arguments.run_name) or header.run_name_decomposed == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]

    header.attack_targeted_attribute = arguments.attribute

    logger.log_trace("WandB run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("Model type: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_decomposed["dir_dataset_test"] + "\".")
    logger.log_trace("Targeted attribute: \"" + str(header.attack_targeted_attribute) + "\".")

    return

def processArgumentsTestBaseline():
    arguments = initializeArgumentsTest()

    header.config_baseline["seed"] = arguments.seed

    if arguments.test_dataset_dir != "":
        header.config_baseline["dir_dataset_test"] = arguments.test_dataset_dir

    if not initializeRunNameBaseline(arguments.run_name) or header.run_name_baseline == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_baseline["model"] = header.run_name_baseline.split(".")[1]

    logger.log_trace("WandB run name: \"" + header.run_name_baseline + "\".")
    logger.log_trace("Model type: \"" + header.config_baseline["model"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_baseline["seed"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_baseline["dir_dataset_test"] + "\".")

    return

def processArgumentsTestDecomposed():
    arguments = initializeArgumentsTest()

    header.config_decomposed["seed"] = arguments.seed

    if arguments.test_dataset_dir != "":
        header.config_decomposed["dir_dataset_test"] = arguments.test_dataset_dir

    if not initializeRunNameDecomposed(arguments.run_name) or header.run_name_decomposed == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]

    logger.log_trace("WandB run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("Model type: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_decomposed["dir_dataset_test"] + "\".")

    return

def processArgumentsTrainBaseline():
    arguments = initializeArgumentsTrain()

    if arguments.model != "":
        header.config_baseline["model"] = arguments.model

    header.config_baseline["data_loader_batch_size"] = arguments.batch_size
    header.config_baseline["epochs"] = arguments.epochs

    if arguments.fine_tune == 0:
        header.config_baseline["fine_tuning"] = False
    else:
        header.config_baseline["fine_tuning"] = True

    header.config_baseline["seed"] = arguments.seed

    resume = initializeRunNameBaseline(arguments.run_name)

    if header.run_name_baseline == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_baseline["model"] = header.run_name_baseline.split(".")[1]

    logger.log_trace("WandB run name: \"" + header.run_name_baseline + "\".")
    logger.log_trace("Model type: \"" + header.config_baseline["model"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_baseline["data_loader_batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_baseline["epochs"]) + ".")
    logger.log_trace("Whether to perform backbone fine-tuning: " + str(header.config_baseline["fine_tuning"]) + ".")
    logger.log_trace("Randomization seed: " + str(header.config_baseline["seed"]) + ".")

    return resume

def processArgumentsTrainDecomposed():
    arguments = initializeArgumentsTrain()

    if arguments.model != "":
        header.config_decomposed["model"] = arguments.model

    header.config_decomposed["data_loader_batch_size"] = arguments.batch_size
    header.config_decomposed["epochs"] = arguments.epochs

    if arguments.fine_tune == 0:
        header.config_decomposed["fine_tuning"] = False
    else:
        header.config_decomposed["fine_tuning"] = True

    if arguments.use_covariance_loss == 0:
        header.config_decomposed["use_covariance_loss"] = False
    else:
        header.config_decomposed["use_covariance_loss"] = True

    header.config_decomposed["seed"] = arguments.seed

    resume = initializeRunNameDecomposed(arguments.run_name)

    if header.run_name_decomposed == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]

    logger.log_trace("WandB run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("Model type: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_decomposed["data_loader_batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_decomposed["epochs"]) + ".")
    logger.log_trace("Whether to perform backbone fine-tuning: " + str(header.config_decomposed["fine_tuning"]) + ".")
    logger.log_trace("Whether to use covariance loss: " + str(header.config_decomposed["use_covariance_loss"]) + ".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")

    return resume

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    width_image = image.shape[1]
    width_resize = image.shape[1]
    height_image = image.shape[0]
    height_resize = image.shape[0]

    if width is None and height is None:
        return image

    if width is None:
        resize_ratio = height / height_image
        width_resize = int(width_image * resize_ratio)
        height_resize = height
    else:
        resize_ratio = width / width_image
        width_resize = width
        height_resize = int(height_image * resize_ratio)

    return cv2.resize(image, (width_resize, height_resize), interpolation=inter)

def saveCheckpoint(dir_checkpoints, file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_schedulers, model, optimizers):
    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok = True)

    learning_rate_scheduler_state_dict_list = []
    optimizer_state_dict_list = []

    for learning_rate_scheduler in learning_rate_schedulers:
        learning_rate_scheduler_state_dict_list.append(learning_rate_scheduler.state_dict())

    for optimizer in optimizers:
        optimizer_state_dict_list.append(optimizer.state_dict())

    checkpoint = {
        "accuracy_validation_best": accuracy_validation_best,
        "batch_step_train": batch_step_train,
        "batch_step_validate": batch_step_validate,
        "criterions": criterions,
        "epoch": epoch,
        "learning_rate_scheduler_state_dict_list": learning_rate_scheduler_state_dict_list,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict_list": optimizer_state_dict_list
    }
    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    torch.save(checkpoint, file_path_checkpoint)

    try:
        wandb.save(file_path_checkpoint, base_path = dir_checkpoints)
    except:
        pass
    else:
        logger.log_info("Saved checkpoint \"" + file_name_checkpoint + "\" to Weights & Biases.")

    logger.log_info("Saved checkpoint \"" + file_name_checkpoint + "\".")

    return

def setSeed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    return

def wAndBDefineMetrics():
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
