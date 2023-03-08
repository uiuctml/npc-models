import multiprocessing
import type

log_level = type.LogLevel.debug
project_name = "mapillary-dataset-network"
run_mode = "online"
run_name_baseline_keyword = "baseline"
run_name_decomposed_keyword = "decomposed"
run_name_baseline = ""
run_name_decomposed = ""

config_baseline = {
    "data_loader_batch_size": 768,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../mapillary-dataset/images/split/original/test",
    "dir_dataset_train": "../../mapillary-dataset/images/split/original/train",
    "dir_dataset_validation": "../../mapillary-dataset/images/split/original/validate",
    "dropout_probability": 0.5,
    "epochs": 100,
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
    "data_loader_batch_size": 768,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dataset_delimiter_file_name": "---",
    "dataset_delimiter_label": "--",
    "dataset_label_undefined_keyword": "undefined",
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../mapillary-dataset/images/split/generated/test",
    "dir_dataset_train": "../../mapillary-dataset/images/split/generated/train",
    "dir_dataset_validation": "../../mapillary-dataset/images/split/generated/validate",
    "dropout_probability": 0.5,
    "epochs": 100,
    "file_name_checkpoint": run_name_decomposed + ".tar",
    "file_name_checkpoint_best": run_name_decomposed + ".best.tar",
    "file_name_config_dataset": "dataset.json",
    "file_name_config_dataset_generation": "generate.json",
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
