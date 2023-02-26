import multiprocessing
import type
import utility

log_level = type.LogLevel.info
project_name = "mapillary-dataset-network"
run_mode = "online"
run_name_baseline = utility.wAndBGenerateRunName("baseline")
run_name_decomposed = utility.wAndBGenerateRunName("decomposed")

config_baseline = {
    "data_loader_batch_size": 32,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dataset_split_percentage_train": 0.8,
    "dataset_split_percentage_validation": 0.1,
    "dataset_split_percentage_test": 0.1,
    "dir_checkpoints": "../checkpoints",
    "dir_dataset": "../../mapillary-dataset/images/sliced/original",
    "dropout_probability": 0.5,
    "epochs": 35,
    "file_name_checkpoint": run_name_baseline + ".tar",
    "file_name_checkpoint_best": run_name_baseline + ".best.tar",
    "fine_tuning": True,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_patient": 1,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 0,
    "learning_rate_scheduler_min_learning_rate": 0,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_verbose": True,
    "l2_lambda": 1e-3,
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
    "model_pretrained_weights": "IMAGENET1K_V2",
    "optimizer_learning_rate": 1e-3,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 1e-6,
    "use_l2_loss": False
}

config_decomposed = {
    "data_loader_batch_size": 32,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dataset_config_file_name": "dataset.json",
    "dataset_delimiter_file_name": "---",
    "dataset_delimiter_label": "--",
    "dataset_split_percentage_train": 0.8,
    "dataset_split_percentage_validation": 0.1,
    "dataset_split_percentage_test": 0.1,
    "dir_checkpoints": "../checkpoints",
    "dir_dataset": "../../mapillary-dataset/images/sliced/generated",
    "dropout_probability": 0.5,
    "epochs": 35,
    "file_name_checkpoint": run_name_decomposed + ".tar",
    "file_name_checkpoint_best": run_name_decomposed + ".best.tar",
    "fine_tuning": True,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_patient": 1,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 0,
    "learning_rate_scheduler_min_learning_rate": 0,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_verbose": True,
    "l2_lambda": 1e-3,
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
    "model_pretrained_weights": "IMAGENET1K_V2",
    "optimizer_learning_rate": 1e-3,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 1e-6,
    "use_l2_loss": False
}
