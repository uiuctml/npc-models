import multiprocessing
import type

composed_find_mpe = False
composed_save_mpe = True
composed_spn_on_cpu = True
counterfactual_implausibility_margin_range = 1e-3
counterfactual_implausibility_margin_sum = 1e-6
counterfactual_learning_rate = 1e-2
counterfactual_qp_epsilon = 1
counterfactual_qp_solver = "clarabel"
counterfactual_save = True
counterfactual_steps = 100
cuda_allow_tf32 = False
dataset_prefix = "awa2"
dir_output_counterfactual = "../output/counterfactual"
dir_output_mpe = "../output/mpe"
file_name_counterfactual = "counterfactual.json"
file_name_mpe = "mpe.json"
file_path_dataset_config = "../../visat-dataset-tools/configs/" + dataset_prefix + ".json"
interpret_delimiter_label = "--"
interpret_dir_dataset = "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/test"
interpret_label_width_attribute = 200
interpret_label_width_original = 350
interpret_mpe = False
interpret_preprocess = False
interpret_threshold_probability = 0.1
interpret_viewer_height = 300
interpret_viewer_width = 300
log_level = type.LogLevel.debug
model_head_hidden_size = 128
project_name = "visat-models"
run_mode = "disabled"
run_name_baseline_keyword = "baseline"
run_name_decomposed_keyword = "decomposed"
run_name_reference_keyword = "reference"
run_name_relation_keyword = "relation"
run_name_spn_keyword = "spn"
run_name_baseline = ""
run_name_decomposed = ""
run_name_reference = ""
run_name_relation = ""
run_name_spn = ""
seed = 42
show_model_summary = False

config_baseline = {
    "data_loader_batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/test",
    "dir_dataset_train": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/train",
    "dir_dataset_validation": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/validate",
    "epochs": 150,
    "file_name_checkpoint": run_name_baseline + ".tar",
    "file_name_checkpoint_best": run_name_baseline + ".best.tar",
    "fine_tuning": True,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_patience": 10,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 0,
    "learning_rate_scheduler_min_learning_rate": 0,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "model": "vit_b_32",
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
    "model_pretrained_weights": "IMAGENET1K_V1",
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 4e-5,
    "run_name": run_name_baseline,
    "seed": seed,
    "type": "baseline",
}

config_decomposed = {
    "data_loader_batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/test",
    "dir_dataset_train": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/train",
    "dir_dataset_validation": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/validate",
    "epochs": 150,
    "file_name_checkpoint": run_name_decomposed + ".tar",
    "file_name_checkpoint_best": run_name_decomposed + ".best.tar",
    "fine_tuning": True,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_patience": 10,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 0,
    "learning_rate_scheduler_min_learning_rate": 0,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_last_epoch": -1,
    "model": "resnet34_mtl",
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
    "model_pretrained_weights": "IMAGENET1K_V1",
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 4e-5,
    "run_name": run_name_decomposed,
    "seed": seed,
    "type": "decomposed",
}

config_reference = {
    "concept_loss_weight": 1,
    "data_loader_batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/test",
    "dir_dataset_train": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/train",
    "dir_dataset_validation": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/validate",
    "epochs": 150,
    "file_name_checkpoint": run_name_reference + ".tar",
    "file_name_checkpoint_best": run_name_reference + ".best.tar",
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_patience": 10,
    "model": "cbm",
    "model_embedding_size": 16,
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 4e-5,
    "run_name": run_name_reference,
    "seed": seed,
    "type": "reference",
}

config_spn = {
    "dir_checkpoints": "../checkpoints",
    "epochs": 50,
    "dir_dataset_test": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/test",
    "dir_dataset_train": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/train",
    "dir_dataset_validation": "../../../datasets/" + dataset_prefix + "-dataset/images/split/original/validate",
    "file_path_spn": "../../learnspn/output/learnspn/" + dataset_prefix + ".spn.txt",
    "file_path_dataset_test_txt": "../../../datasets/" + dataset_prefix + "-dataset/images/split/spn/test.txt",
    "file_name_checkpoint": run_name_spn + ".tar",
    "file_name_checkpoint_best": run_name_spn + ".best.tar",
    "fine_tuning": False,
    "epsilon_projection": 1e-2,
    "epsilon_smoothing": 1e-3,
    "growth_threshold": 100,
    "joint_inference_only": True,
    "learning_rate_scheduler_factor": 0.5,
    "learning_rate_scheduler_patience": 1,
    "learning_rate_scheduler_threshold": 1e-2,
    "learning_rate_scheduler_cooldown": 1,
    "learning_rate_scheduler_min_learning_rate": 1e-4,
    "model_pretrained_weights": "",
    "optimizer": "cccp_generative",
    "optimizer_learning_rate": 1e-1,
    "optimizer_prior_factor": 1e2,
    "randomize_weights": False,
    "run_name": run_name_spn,
    "seed": seed,
    "stopping_criterion": 1e-4,
    "test_on_txt": True,
    "type": "spn",
}
