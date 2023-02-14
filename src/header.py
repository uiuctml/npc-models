import multiprocessing
import type

baseline_data_loader_batch_size = 32
baseline_data_loader_shuffle = True
baseline_data_loader_worker_count = multiprocessing.cpu_count()
baseline_dropout_probability = 0.5
baseline_dry_run = False
baseline_epochs = 35
baseline_fine_tuning = True
baseline_learning_rate_scheduler_mode = "min"
baseline_learning_rate_scheduler_factor = 0.1
baseline_learning_rate_scheduler_patient = 1
baseline_learning_rate_scheduler_threshold = 1e-4
baseline_learning_rate_scheduler_threshold_mode = "rel"
baseline_learning_rate_scheduler_cooldown = 0
baseline_learning_rate_scheduler_min_learning_rate = 0
baseline_learning_rate_scheduler_min_learning_rate_decay = 1e-8
baseline_learning_rate_scheduler_verbose = True
baseline_load_best = False
baseline_l2_lambda = 1e-3
baseline_model_dir = "../model"
baseline_model_dir_best = "../model"
baseline_model_input_height = 224
baseline_model_input_width = baseline_model_input_height
baseline_model_input_channels = 3
baseline_model_pretrained_weights = "IMAGENET1K_V2"
baseline_network_revision = type.NetworkRevision.revision_0
baseline_optimizer_learning_rate = 1e-3
baseline_optimizer_momentum = 0.9
baseline_optimizer_weight_decay = 1e-6
baseline_use_l2_loss = False

dataset_dir = "../../mapillary-dataset"
dataset_dir_split = dataset_dir + "/images" + "/split" + "/original"
dataset_dir_train_validation = dataset_dir_split + "/train_val"
dataset_dir_test = dataset_dir_split + "/test"
dataset_split_percentage_train = 0.8
dataset_split_percentage_validation = 1 - dataset_split_percentage_train
dataset_view_row_count = 5

evaluate_label_excluded = set([
    "information--stairs--g1",
    "warning--restricted-zone--g1",
    "information--no-parking--g3",
    "regulatory--weight-limit-with-trucks--g1",
    "regulatory--no-right-turn--g3",
    "regulatory--truck-speed-limit-60--g1",
    "warning--shared-lane-motorcycles-bicycles--g1",
    "regulatory--end-of-maximum-speed-limit-70--g1",
    "information--central-lane--g1",
    "information--dead-end-except-bicycles--g1"
])
evaluate_model_dir = "../model"

model_file_name_accuracy_validation = "accuracy_validation.pth"
model_file_name_class_indices_test = "class_indices_test.pth"
model_file_name_classes = "classes.pth"
model_file_name_criterion = "criterion.pth"
model_file_name_data_loader_train = "data_loader_train.pth"
model_file_name_data_loader_validation = "data_loader_validation.pth"
model_file_name_epoch = "epoch.pth"
model_file_name_learning_rate_scheduler = "learning_rate_scheduler.pth"
model_file_name_model = "model.pth"
model_file_name_model_best = "model_best.pth"
model_file_name_optimizer = "optimizer.pth"
model_file_name_outputs_test = "outputs_test.pth"
model_file_name_statistics_evaluate = "statistics_evaluate.json"
model_file_name_statistics_test = "statistics_test.json"
model_file_name_statistics_train = "statistics_train.json"

log_level = type.LogLevel.info

plot_model_dir = "../model"
plot_save_evaluate = True
plot_save_test = True
plot_save_train = True
plot_show_evaluate = True
plot_show_test = True
plot_show_train = True
plot_subplot_count_evaluate = 9
plot_subplot_col_count_evaluate = 3
