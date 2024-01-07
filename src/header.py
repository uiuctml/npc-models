import multiprocessing
import type

cuda_allow_tf32 = True
dataset_config_file_path = "../../mapillary-dataset-tools/config/visat.json"
file_path_spn_matrix_a = ""
log_level = type.LogLevel.debug
project_name = "mapillary-dataset-network"
run_mode = "disabled"
run_name_baseline_keyword = "baseline"
run_name_decomposed_keyword = "decomposed"
run_name_baseline = ""
run_name_decomposed = ""
seed = 42
show_model_summary = False

config_baseline = {
    "data_loader_batch_size": 512,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../mapillary-dataset/images/split/original/test",
    "dir_dataset_train": "../../mapillary-dataset/images/split/original/train",
    "dir_dataset_validation": "../../mapillary-dataset/images/split/original/validate",
    "dir_test_output": "../output/test",
    "dropout_probability": 0.5,
    "epochs": 100,
    "file_name_checkpoint": run_name_baseline + ".tar",
    "file_name_checkpoint_best": run_name_baseline + ".best.tar",
    "file_name_test_output_clean": "test_output_baseline_clean.txt",
    "file_name_test_output_attacked": "test_output_baseline_attacked.txt",
    "file_path_input_adversarial": "1e-2-50.pt",
    "fine_tuning": True,
    "input_grayscale": False,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.5,
    "learning_rate_scheduler_patience": 3,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 3,
    "learning_rate_scheduler_min_learning_rate": 1e-5,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_verbose": True,
    "log_test_output": False,
    "l2_lambda": 1e-3,
    "model": "vit_b_32",
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
    "model_pretrained_weights": "IMAGENET1K_V1",
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 1e-6,
    "run_name": run_name_baseline,
    "seed": seed,
    "test_adversarial": False,
    "type": "baseline",
    "use_l2_loss": False
}

config_decomposed = {
    "data_loader_batch_size": 512,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dataset_delimiter_file_name": "---",
    "dataset_delimiter_label": "--",
    "dataset_label_undefined_keyword": "undefined",
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../mapillary-dataset/images/split/original/test",
    "dir_dataset_train": "../../mapillary-dataset/images/split/original/train",
    "dir_dataset_validation": "../../mapillary-dataset/images/split/original/validate",
    "dir_test_output": "../output/test",
    "dropout_probability": 0.5,
    "epochs": 100,
    "factor_loss_covariance": 1e-8,
    "file_name_checkpoint": run_name_decomposed + ".tar",
    "file_name_checkpoint_best": run_name_decomposed + ".best.tar",
    "file_name_test_output_clean": "test_output_decomposed_clean.txt",
    "file_name_test_output_attacked": "test_output_decomposed_attacked.txt",
    "file_path_input_adversarial": "1e-2-50.pt",
    "fine_tuning": True,
    "head_hidden_sizes": {
        "color": 128,
        "shape": 128,
        "symbol": 128,
        "text": 128
    },
    "input_grayscale": False,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.5,
    "learning_rate_scheduler_patience": 3,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 3,
    "learning_rate_scheduler_min_learning_rate": 1e-5,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_verbose": True,
    "log_test_output": False,
    "l2_lambda": 1e-3,
    "model": "vit_b_32_mtl",
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
    "model_pretrained_weights": "IMAGENET1K_V1",
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 1e-6,
    "run_name": run_name_decomposed,
    "seed": seed,
    "test_adversarial": False,
    "type": "decomposed",
    "use_l2_loss": False,
    "use_covariance_loss": False
}

visualize_dir_output = "../output/visualize"
visualize_dir_output_correct = visualize_dir_output + "/correct"
visualize_dir_output_incorrect = visualize_dir_output + "/incorrect"
visualize_save = False
visualize_show = False
visualize_size = 900
visualize_text_position_y_start = 30
visualize_text_position_y_increment = 20
visualize_text_scale = 0.5
visualize_text_thickness = 2
