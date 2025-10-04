import multiprocessing
import type

cuda_allow_tf32 = False
dataset_prefix = "awa2"
dataset_config_file_path = "../../npc-dataset-utils/configs/npc-dataset-utils/" + dataset_prefix + ".json"
log_level = type.LogLevel.debug
model_head_hidden_size = 128
project_name = "npc-models"
seed = 42

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

run_mode = "disabled"
run_name_baseline = ""
run_name_decomposed = ""
run_name_reference = ""
run_name_spn = ""
run_name_keyword_baseline = "baseline"
run_name_keyword_decomposed = "decomposed"
run_name_keyword_reference = "reference"
run_name_keyword_spn = "spn"

config_baseline = {
    "batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../datasets/" + dataset_prefix + "/splits/instances/test",
    "dir_dataset_train": "../../datasets/" + dataset_prefix + "/splits/instances/train",
    "dir_dataset_validation": "../../datasets/" + dataset_prefix + "/splits/instances/validate",
    "epochs": 150,
    "file_name_checkpoint": run_name_baseline + ".tar",
    "file_name_checkpoint_best": run_name_baseline + ".best.tar",
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
    "run_name": run_name_baseline,
    "seed": seed,
    "type": "baseline",
}

config_decomposed = {
    "batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../datasets/" + dataset_prefix + "/splits/instances/test",
    "dir_dataset_train": "../../datasets/" + dataset_prefix + "/splits/instances/train",
    "dir_dataset_validation": "../../datasets/" + dataset_prefix + "/splits/instances/validate",
    "epochs": 150,
    "file_name_checkpoint": run_name_decomposed + ".tar",
    "file_name_checkpoint_best": run_name_decomposed + ".best.tar",
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_patience": 10,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 0,
    "learning_rate_scheduler_min_learning_rate": 0,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_last_epoch": -1,
    "model_input_height": 224,
    "model_input_width": 224,
    "model_input_channels": 3,
    "model_pretrained_weights": "",
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 4e-5,
    "run_name": run_name_decomposed,
    "seed": seed,
    "type": "decomposed",
}

config_reference = {
    "concept_loss_weight": 1,
    "batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints": "../checkpoints",
    "dir_dataset_test": "../../datasets/" + dataset_prefix + "/splits/instances/test",
    "dir_dataset_train": "../../datasets/" + dataset_prefix + "/splits/instances/train",
    "dir_dataset_validation": "../../datasets/" + dataset_prefix + "/splits/instances/validate",
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
    "dir_dataset_test": "../../datasets/" + dataset_prefix + "/splits/pc/test.txt",
    "dir_dataset_train": "../../datasets/" + dataset_prefix + "/splits/pc/train.txt",
    "dir_dataset_validation": "../../datasets/" + dataset_prefix + "/splits/pc/validate.txt",
    "file_path_spn": "../../learnspn/output/learnspn/" + dataset_prefix + ".spn.txt",
    "file_name_checkpoint": run_name_spn + ".tar",
    "file_name_checkpoint_best": run_name_spn + ".best.tar",
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
    "run_name": run_name_spn,
    "seed": seed,
    "stopping_criterion": 1e-4,
    "type": "spn",
}
