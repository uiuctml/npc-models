import multiprocessing
import os
import type

project_name = "npc-models"
dir_outputs = os.path.join("../outputs", project_name)
checkpoint_dir = os.path.join(dir_outputs, "checkpoints")
checkpoint_postfix = ".zip"
checkpoint_postfix_best = ".best.zip"
cuda_allow_tf32 = False
dataset_prefix = "awa2"
dataset_config_file_path = os.path.join("../../npc-dataset-utils/configs/npc-dataset-utils", dataset_prefix + ".json")
dataset_dir = os.path.join("../../datasets", dataset_prefix)
log_level = type.LogLevel.info
run_mode = "disabled"

composed_mpe_dir_outputs = os.path.join(dir_outputs, "mpe")
composed_mpe_file_name = "mpe.json"
composed_mpe_find = False
composed_mpe_save = True
composed_spn_on_cpu = True

counterfactual_dir_outputs = os.path.join(dir_outputs, "counterfactual")
counterfactual_file_name = "counterfactual.json"
counterfactual_learning_rate = 5e-2
counterfactual_save = True
counterfactual_steps = 100

interpret_dir_dataset = os.path.join(dataset_dir, "splits/instances/test")
interpret_label_width_attribute = 200
interpret_label_width_original = 350
interpret_mpe = False
interpret_preprocess = False
interpret_viewer_height = 300
interpret_viewer_width = 300

config_baseline = {
    "batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_dataset_test": os.path.join(dataset_dir, "splits/instances/test"),
    "dir_dataset_train": os.path.join(dataset_dir, "splits/instances/train"),
    "dir_dataset_validation": os.path.join(dataset_dir, "splits/instances/validate"),
    "epochs": 150,
    "file_name_checkpoint": "",
    "file_name_checkpoint_best": "",
    "learning_rate_scheduler_cooldown": 0,
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_min_learning_rate": 0,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_patience": 10,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "model_input_height": 224,
    "model_input_width": 224,
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 4e-5,
    "run_name": "",
    "seed": 42,
    "type": "baseline",
}

config_decomposed = {
    "batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_dataset_test": os.path.join(dataset_dir, "splits/instances/test"),
    "dir_dataset_train": os.path.join(dataset_dir, "splits/instances/train"),
    "dir_dataset_validation": os.path.join(dataset_dir, "splits/instances/validate"),
    "epochs": 150,
    "file_name_checkpoint": "",
    "file_name_checkpoint_best": "",
    "learning_rate_scheduler_cooldown": 0,
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_min_learning_rate": 0,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_patience": 10,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "model_head_hidden_size": 128,
    "model_input_height": 224,
    "model_input_width": 224,
    "model_pretrained_weights": "",
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 4e-5,
    "run_name": "",
    "seed": 42,
    "type": "decomposed",
}

config_reference = {
    "batch_size": 256,
    "concept_loss_weight": 1,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_dataset_test": os.path.join(dataset_dir, "splits/instances/test"),
    "dir_dataset_train": os.path.join(dataset_dir, "splits/instances/train"),
    "dir_dataset_validation": os.path.join(dataset_dir, "splits/instances/validate"),
    "epochs": 150,
    "file_name_checkpoint": "",
    "file_name_checkpoint_best": "",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_patience": 10,
    "model": "cbm",
    "model_embedding_size": 16,
    "model_input_height": 224,
    "model_input_width": 224,
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 4e-5,
    "run_name": "",
    "seed": 42,
    "type": "reference",
}

config_spn = {
    "epochs": 50,
    "file_name_checkpoint": "",
    "file_name_checkpoint_best": "",
    "file_path_dataset_test": os.path.join(dataset_dir, "splits/pc/test.txt"),
    "file_path_dataset_train": os.path.join(dataset_dir, "splits/pc/train.txt"),
    "file_path_dataset_validation": os.path.join(dataset_dir, "splits/pc/validate.txt"),
    "file_path_spn": "../../learnspn/outputs/learnspn/" + dataset_prefix + ".spn.txt",
    "epsilon_projection": 1e-2,
    "epsilon_smoothing": 1e-3,
    "inference_only": True,
    "learning_rate_scheduler_cooldown": 1,
    "learning_rate_scheduler_factor": 0.5,
    "learning_rate_scheduler_min_learning_rate": 1e-4,
    "learning_rate_scheduler_patience": 1,
    "learning_rate_scheduler_threshold": 1e-2,
    "model_pretrained_weights": "",
    "optimizer_learning_rate": 1e-1,
    "optimizer_prior_factor": 1e2,
    "randomize_weights": False,
    "run_name": "",
    "seed": 42,
    "stopping_criterion": 1e-4,
    "type": "spn",
}
