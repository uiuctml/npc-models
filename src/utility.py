import argparse
import cv2
import datetime
import header
import logger
import numpy
import os
import random
import socket
import torch
import torchvision
import wandb

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

def initializeArgumentsInference():
    parser = argparse.ArgumentParser()

    parser.add_argument("-rb", "--run-name-baseline", type = str, default = "", help = "Baseline WandB run name.", required = True)
    parser.add_argument("-rd", "--run-name-decomposed", type = str, default = "", help = "Decomposed WandB run name.", required = True)
    parser.add_argument("-ib", "--adversarial-input-file-baseline", type = str, default = "", help = "Baseline path to adversarial input file.")
    parser.add_argument("-id", "--adversarial-input-file-decomposed", type = str, default = "", help = "Decomposed path to adversarial input file.")
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")
    parser.add_argument("-m", "--spn-matrix-a-file", type = str, default = "", help = "Path to SPN matrix A file.")
    parser.add_argument("-a", "--test-adversarial", type = int, default = 0, help = "Whether to perform adversarial tests.")
    parser.add_argument("-db", "--test-dataset-dir-baseline", type = str, default = "", help = "Baseline directory of dataset testing split.")
    parser.add_argument("-dd", "--test-dataset-dir-decomposed", type = str, default = "", help = "Decomposed directory of dataset testing split.")

    return parser.parse_args()

def initializeArgumentsTest():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "WandB run name.", required = True)
    parser.add_argument("-i", "--adversarial-input-file", type = str, default = "", help = "Path to adversarial input file.")
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")
    parser.add_argument("-a", "--test-adversarial", type = int, default = 0, help = "Whether to perform adversarial tests.")
    parser.add_argument("-d", "--test-dataset-dir", type = str, default = "", help = "Directory of dataset testing split.")

    return parser.parse_args()

def initializeArgumentsTrain():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "WandB run name.")
    parser.add_argument("-f", "--fine-tune", type = int, default = 1, help = "Whether to perform backbone fine-tuning.")
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")

    return parser.parse_args()

def initializeRunNameBaseline(run_name):
    resume = False
    if run_name != "":
        if run_name.split(".")[0] != header.run_name_baseline_keyword:
            logger.log_fatal("Invalid baseline run name. Quit.")
            exit(1)

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
            exit(1)

        header.run_name_decomposed = run_name
        resume = True
    else:
        header.run_name_decomposed = wAndBGenerateRunName(header.run_name_decomposed_keyword, header.config_decomposed["model"], header.config_decomposed["seed"], header.config_decomposed["fine_tuning"])

    header.config_decomposed["file_name_checkpoint"] = header.run_name_decomposed + ".tar"
    header.config_decomposed["file_name_checkpoint_best"] = header.run_name_decomposed + ".best.tar"
    header.config_decomposed["run_name"] = header.run_name_decomposed

    return resume

def loadCheckpoint(dir_checkpoints, file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterion, epoch, learning_rate_scheduler, model, optimizer):
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
            criterion = checkpoint["criterion"]
            epoch = checkpoint["epoch"]
            learning_rate_scheduler.load_state_dict(checkpoint["learning_rate_scheduler_state_dict"])
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
        else:
            logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
            exit(1)

    return (accuracy_validation_best, batch_step_train, batch_step_validate, criterion, epoch)

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
        exit(1)

    return

def logTestOutput(config, output):
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

    if config["test_adversarial"] == False and dir_dataset_test.split('/')[-1] == "test":
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

    if config["test_adversarial"] == False:
        file_line += dir_dataset_test.split('/')[-1]
    else:
        file_line += config["file_path_input_adversarial"].split("/")[-1].split(".best.tar.pt")[0]

    for value in output:
        file_line += "\t"
        file_line += str(value)

    file_line += "\n"

    with open(file_path_test_output, "a") as file_test_output:
        if file_empty:
            if config["type"] == "baseline":
                file_test_output.write("# seed\tmodel\tattack\taccuracy\tprecision\trecall\n")
            elif config["type"] == "decomposed":
                file_test_output.write("# seed\tmodel\tattack\taccuracy_color\taccuracy_shape\taccuracy_symbol\taccuracy_text\tprecision_color\tprecision_shape\tprecision_symbol\tprecision_text\trecall_color\trecall_shape\trecall_symbol\trecall_text\n")
            else:
                logger.log_fatal("Unknown configuration type")
                exit(1)

        file_test_output.write(file_line)

    logger.log_info("Logged test output to \"" + file_path_test_output + "\".")

    return

def processArgumentsInference():
    arguments = initializeArgumentsInference()

    if arguments.adversarial_input_file_baseline != "":
        header.config_baseline["file_path_input_adversarial"] = arguments.adversarial_input_file_baseline

    if arguments.adversarial_input_file_decomposed != "":
        header.config_decomposed["file_path_input_adversarial"] = arguments.adversarial_input_file_decomposed

    header.config_baseline["seed"] = arguments.seed
    header.config_decomposed["seed"] = arguments.seed

    if arguments.spn_matrix_a_file != "":
        header.file_path_spn_matrix_a = arguments.spn_matrix_a_file

    if arguments.test_adversarial == 0:
        header.config_baseline["test_adversarial"] = False
        header.config_decomposed["test_adversarial"] = False
    else:
        header.config_baseline["test_adversarial"] = True
        header.config_decomposed["test_adversarial"] = True

    if arguments.test_dataset_dir_baseline != "":
        header.config_baseline["dir_dataset_test"] = arguments.test_dataset_dir_baseline

    if arguments.test_dataset_dir_decomposed != "":
        header.config_decomposed["dir_dataset_test"] = arguments.test_dataset_dir_decomposed

    if not initializeRunNameBaseline(arguments.run_name_baseline) or header.run_name_baseline == "":
        logger.log_fatal("Baseline run name missing. Quit.")
        exit(1)

    if not initializeRunNameDecomposed(arguments.run_name_decomposed) or header.run_name_decomposed == "":
        logger.log_fatal("Baseline run name missing. Quit.")
        exit(1)

    header.config_baseline["model"] = header.run_name_baseline.split(".")[1]
    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]

    logger.log_trace("Baseline WandB run name: \"" + header.run_name_baseline + "\".")
    logger.log_trace("Baseline path to adversarial input file: \"" + header.config_baseline["file_path_input_adversarial"] + "\".")
    logger.log_trace("Baseline directory of dataset testing split: \"" + header.config_baseline["dir_dataset_test"] + "\".")
    logger.log_trace("Baseline model type: \"" + header.config_baseline["model"] + "\".")
    logger.log_trace("Decomposed WandB run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("Decomposed path to adversarial input file: \"" + header.config_decomposed["file_path_input_adversarial"] + "\".")
    logger.log_trace("Decomposed directory of dataset testing split: \"" + header.config_decomposed["dir_dataset_test"] + "\".")
    logger.log_trace("Decomposed model type: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("Path to SPN matrix A file: " + header.file_path_spn_matrix_a + ".")
    logger.log_trace("Randomization seed: " + str(header.config_baseline["seed"]) + ".")
    logger.log_trace("Whether to perform adversarial tests: " + str(header.config_baseline["test_adversarial"]) + ".")

    return

def processArgumentsTrainBaseline():
    arguments = initializeArgumentsTrain()

    if arguments.fine_tune == 0:
        header.config_baseline["fine_tuning"] = False
    else:
        header.config_baseline["fine_tuning"] = True

    header.config_baseline["seed"] = arguments.seed

    resume = initializeRunNameBaseline(arguments.run_name)

    if header.run_name_baseline == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(1)

    logger.log_trace("WandB run name: \"" + header.run_name_baseline + "\".")
    logger.log_trace("Whether to perform backbone fine-tuning: " + str(header.config_baseline["fine_tuning"]) + ".")
    logger.log_trace("Randomization seed: " + str(header.config_baseline["seed"]) + ".")

    return resume

def processArgumentsTrainDecomposed():
    arguments = initializeArgumentsTrain()

    if arguments.fine_tune == 0:
        header.config_decomposed["fine_tuning"] = False
    else:
        header.config_decomposed["fine_tuning"] = True

    header.config_decomposed["seed"] = arguments.seed

    resume = initializeRunNameDecomposed(arguments.run_name)

    if header.run_name_decomposed == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(1)

    logger.log_trace("WandB run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("Whether to perform backbone fine-tuning: " + str(header.config_decomposed["fine_tuning"]) + ".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")

    return resume

def processArgumentsTestBaseline():
    arguments = initializeArgumentsTest()

    if arguments.adversarial_input_file != "":
        header.config_baseline["file_path_input_adversarial"] = arguments.adversarial_input_file

    header.config_baseline["seed"] = arguments.seed

    if arguments.test_adversarial == 0:
        header.config_baseline["test_adversarial"] = False
    else:
        header.config_baseline["test_adversarial"] = True

    if arguments.test_dataset_dir != "":
        header.config_baseline["dir_dataset_test"] = arguments.test_dataset_dir

    if not initializeRunNameBaseline(arguments.run_name) or header.run_name_baseline == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(1)

    header.config_baseline["model"] = header.run_name_baseline.split(".")[1]

    logger.log_trace("WandB run name: \"" + header.run_name_baseline + "\".")
    logger.log_trace("Path to adversarial input file: \"" + header.config_baseline["file_path_input_adversarial"] + "\".")
    logger.log_trace("Model type: \"" + header.config_baseline["model"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_baseline["seed"]) + ".")
    logger.log_trace("Whether to perform adversarial tests: " + str(header.config_baseline["test_adversarial"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_baseline["dir_dataset_test"] + "\".")

    return

def processArgumentsTestDecomposed():
    arguments = initializeArgumentsTest()

    if arguments.adversarial_input_file != "":
        header.config_decomposed["file_path_input_adversarial"] = arguments.adversarial_input_file

    header.config_decomposed["seed"] = arguments.seed

    if arguments.test_adversarial == 0:
        header.config_decomposed["test_adversarial"] = False
    else:
        header.config_decomposed["test_adversarial"] = True

    if arguments.test_dataset_dir != "":
        header.config_decomposed["dir_dataset_test"] = arguments.test_dataset_dir

    if not initializeRunNameDecomposed(arguments.run_name) or header.run_name_decomposed == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(1)

    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]

    logger.log_trace("WandB run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("Path to adversarial input file: \"" + header.config_decomposed["file_path_input_adversarial"] + "\".")
    logger.log_trace("Model type: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")
    logger.log_trace("Whether to perform adversarial tests: " + str(header.config_decomposed["test_adversarial"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_decomposed["dir_dataset_test"] + "\".")

    return

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

def saveCheckpoint(dir_checkpoints, file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterion, epoch, learning_rate_scheduler, model, optimizer):
    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok = True)

    checkpoint = {
        "accuracy_validation_best": accuracy_validation_best,
        "batch_step_train": batch_step_train,
        "batch_step_validate": batch_step_validate,
        "criterion": criterion,
        "epoch": epoch,
        "learning_rate_scheduler_state_dict": learning_rate_scheduler.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
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
