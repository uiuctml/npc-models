import multiprocessing
import type

cuda_allow_tf32 = False
checkpoint_dir = "../checkpoints"
checkpoint_postfix = ".zip"
checkpoint_postfix_best = ".best.zip"
dataset_prefix = "awa2"
dataset_config_file_path = "../../npc-dataset-utils/configs/npc-dataset-utils/" + dataset_prefix + ".json"
log_level = type.LogLevel.debug
project_name = "npc-models"
run_mode = "disabled"

composed_mpe_dir_output = "../output/mpe"
composed_mpe_file_name = "mpe.json"
composed_mpe_find = False
composed_mpe_save = True
composed_spn_on_cpu = True

counterfactual_dir_output = "../output/counterfactual"
counterfactual_file_name = "counterfactual.json"
counterfactual_learning_rate = 5e-2
counterfactual_save = True
counterfactual_steps = 100

interpret_dir_dataset = "../../datasets/" + dataset_prefix + "/splits/instances/test"
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
    "dir_dataset_test": "../../datasets/" + dataset_prefix + "/splits/instances/test",
    "dir_dataset_train": "../../datasets/" + dataset_prefix + "/splits/instances/train",
    "dir_dataset_validation": "../../datasets/" + dataset_prefix + "/splits/instances/validate",
    "epochs": 150,
    "file_name_checkpoint": "",
    "file_name_checkpoint_best": "",
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_patience": 10,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 0,
    "learning_rate_scheduler_min_learning_rate": 0,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
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
    "dir_dataset_test": "../../datasets/" + dataset_prefix + "/splits/instances/test",
    "dir_dataset_train": "../../datasets/" + dataset_prefix + "/splits/instances/train",
    "dir_dataset_validation": "../../datasets/" + dataset_prefix + "/splits/instances/validate",
    "epochs": 150,
    "file_name_checkpoint": "",
    "file_name_checkpoint_best": "",
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_patience": 10,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 0,
    "learning_rate_scheduler_min_learning_rate": 0,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_last_epoch": -1,
    "model_head_hidden_size": 128,
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
    "model_pretrained_weights": "",
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 4e-5,
    "run_name": "",
    "seed": 42,
    "type": "decomposed",
}

config_reference = {
    "concept_loss_weight": 1,
    "batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_dataset_test": "../../datasets/" + dataset_prefix + "/splits/instances/test",
    "dir_dataset_train": "../../datasets/" + dataset_prefix + "/splits/instances/train",
    "dir_dataset_validation": "../../datasets/" + dataset_prefix + "/splits/instances/validate",
    "epochs": 150,
    "file_name_checkpoint": "",
    "file_name_checkpoint_best": "",
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
    "run_name": "",
    "seed": 42,
    "type": "reference",
}

config_spn = {
    "dir_dataset_test": "../../datasets/" + dataset_prefix + "/splits/pc/test.txt",
    "dir_dataset_train": "../../datasets/" + dataset_prefix + "/splits/pc/train.txt",
    "dir_dataset_validation": "../../datasets/" + dataset_prefix + "/splits/pc/validate.txt",
    "epochs": 50,
    "file_path_spn": "../../learnspn/output/learnspn/" + dataset_prefix + ".spn.txt",
    "file_name_checkpoint": "",
    "file_name_checkpoint_best": "",
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
    "optimizer_learning_rate": 1e-1,
    "optimizer_prior_factor": 1e2,
    "randomize_weights": False,
    "run_name": "",
    "seed": 42,
    "stopping_criterion": 1e-4,
    "type": "spn",
}
