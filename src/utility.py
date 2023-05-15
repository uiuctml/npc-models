
import cv2
import datetime
import header
import logger
import os
import socket
import torch
import wandb

def computeL2Norm(parameters):
    parameters_list = []

    for parameter in parameters:
        parameters_list.append(parameter.view(-1))

    return torch.square(torch.cat(parameters_list)).sum().item()

def wAndBGenerateRunName(model_name, model_type):
    date_time_list = list(datetime.datetime.now().timetuple())[:-4]
    run_name = model_name
    run_name += "."
    run_name += model_type

    for entry in date_time_list:
        run_name += "."
        run_name += str(entry)

    run_name += "."
    run_name += socket.gethostname()

    return run_name

def initializeRunNameBaseline(run_name):
    resume = False

    if run_name != "":
        if run_name.split(".")[0] != header.run_name_baseline_keyword:
            logger.log_error("Invalid baseline run name. Quit.")
            exit(1)

        header.run_name_baseline = run_name
        resume = True
    else:
        header.run_name_baseline = wAndBGenerateRunName(header.run_name_baseline_keyword, header.config_baseline["model"])

    header.config_baseline["file_name_checkpoint"] = header.run_name_baseline + ".tar"
    header.config_baseline["file_name_checkpoint_best"] = header.run_name_baseline + ".best.tar"

    return resume

def initializeRunNameDecomposed(run_name):
    resume = False

    if run_name != "":
        if run_name.split(".")[0] != header.run_name_decomposed_keyword:
            logger.log_error("Invalid decomposed run name. Quit.")
            exit(1)

        header.run_name_decomposed = run_name
        resume = True
    else:
        header.run_name_decomposed = wAndBGenerateRunName(header.run_name_decomposed_keyword, header.config_decomposed["model"])

    header.config_decomposed["file_name_checkpoint"] = header.run_name_decomposed + ".tar"
    header.config_decomposed["file_name_checkpoint_best"] = header.run_name_decomposed + ".best.tar"

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
