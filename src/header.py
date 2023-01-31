import logger
import multiprocessing

dataset_dir = "../../mapillary-dataset"
dataset_dir_train_validation = dataset_dir + "/images" + "/split" + "/train_val"
dataset_dir_train_test = dataset_dir + "/images" + "/split" + "/test"
dataset_split_percentage_train = 0.8
dataset_split_percentage_validation = 1 - dataset_split_percentage_train
dataset_view_row_count = 5

evaluate_model_dir = "../model"
evaluate_model_file_path_accuracy_validation = evaluate_model_dir + "/accuracy_validation.pth"
evaluate_model_file_path_criterion = evaluate_model_dir + "/criterion.pth"
evaluate_model_file_path_data_loader_train = evaluate_model_dir + "/data_loader_train.pth"
evaluate_model_file_path_data_loader_validation = evaluate_model_dir + "/data_loader_validation.pth"
evaluate_model_file_path_epoch = evaluate_model_dir + "/epoch.pth"
evaluate_model_file_path_learning_rate_scheduler = evaluate_model_dir + "/learning_rate_scheduler.pth"
evaluate_model_file_path_model = evaluate_model_dir + "/model.pth"
evaluate_model_file_path_model = evaluate_model_dir + "/model.pth"
evaluate_model_file_path_optimizer = evaluate_model_dir + "/optimizer.pth"
evaluate_model_file_path_statistics = evaluate_model_dir + "/statistics.json"

log_level = logger.LogLevel.info

train_data_loader_batch_size = 32
train_data_loader_worker_count = multiprocessing.cpu_count()
train_data_loader_shuffle = True
train_dry_run = False
train_epochs = 25
train_learning_rate_scheduler_mode = "min"
train_learning_rate_scheduler_factor = 0.1
train_learning_rate_scheduler_patient = 2
train_learning_rate_scheduler_threshold = 1e-4
train_learning_rate_scheduler_threshold_mode = "abs"
train_learning_rate_scheduler_cooldown = 0
train_learning_rate_scheduler_min_learning_rate = 1e-6
train_learning_rate_scheduler_min_learning_rate_decay = 1e-8
train_learning_rate_scheduler_verbose = True
train_model_dir = "../model"
train_model_file_path_accuracy_validation = train_model_dir + "/accuracy_validation.pth"
train_model_file_path_criterion = train_model_dir + "/criterion.pth"
train_model_file_path_data_loader_train = train_model_dir + "/data_loader_train.pth"
train_model_file_path_data_loader_validation = train_model_dir + "/data_loader_validation.pth"
train_model_file_path_epoch = train_model_dir + "/epoch.pth"
train_model_file_path_learning_rate_scheduler = train_model_dir + "/learning_rate_scheduler.pth"
train_model_file_path_model = train_model_dir + "/model.pth"
train_model_file_path_optimizer = train_model_dir + "/optimizer.pth"
train_model_file_path_statistics = train_model_dir + "/statistics.json"
train_model_input_height = 224
train_model_input_width = train_model_input_height
train_model_input_channels = 3
train_model_pretrained_weights = "IMAGENET1K_V2"
train_optimizer_learning_rate = 0.001
train_optimizer_momentum = 0.9
