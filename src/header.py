import logger
import multiprocessing

dataset_dir = "/data/common/mapillary-dataset"
dataset_dir_train_validation = dataset_dir + "/images" + "/split" + "/train_val"
dataset_dir_test = dataset_dir + "/images" + "/split" + "/test"
dataset_split_percentage_train = 0.8
dataset_split_percentage_validation = 1 - dataset_split_percentage_train
dataset_view_row_count = 5

evaluate_model_dir = "../model"
evaluate_save_plot = True
evaluate_show_plot = False

model_file_name_accuracy_validation = "accuracy_validation.pth"
model_file_name_criterion = "criterion.pth"
model_file_name_data_loader_train = "data_loader_train.pth"
model_file_name_data_loader_validation = "data_loader_validation.pth"
model_file_name_epoch = "epoch.pth"
model_file_name_learning_rate_scheduler = "learning_rate_scheduler.pth"
model_file_name_model = "model.pth"
model_file_name_optimizer = "optimizer.pth"
model_file_name_statistics = "statistics.json"

log_level = logger.LogLevel.info

train_data_loader_batch_size = 32
train_data_loader_shuffle = True
train_data_loader_worker_count = multiprocessing.cpu_count()
train_dropout_probability = 0.5
train_dry_run = False
train_epochs = 35
train_learning_rate_scheduler_mode = "min"
train_learning_rate_scheduler_factor = 0.1
train_learning_rate_scheduler_patient = 1
train_learning_rate_scheduler_threshold = 1e-4
train_learning_rate_scheduler_threshold_mode = "abs"
train_learning_rate_scheduler_cooldown = 0
train_learning_rate_scheduler_min_learning_rate = 1e-6
train_learning_rate_scheduler_min_learning_rate_decay = 1e-8
train_learning_rate_scheduler_verbose = True
train_l2_lambda = 1e-3
train_model_dir = "../model"
train_model_input_height = 224
train_model_input_width = train_model_input_height
train_model_input_channels = 3
train_model_pretrained_weights = "IMAGENET1K_V2"
train_optimizer_learning_rate = 0.001
train_optimizer_momentum = 0.9
train_optimizer_weight_decay = 1e-5
train_use_l2_loss = False

test_data_loader_batch_size = train_data_loader_batch_size
test_data_loader_shuffle = train_data_loader_shuffle
test_data_loader_worker_count = multiprocessing.cpu_count()
test_model_dir = "../model"
test_model_input_height = train_model_input_height
test_model_input_width = train_model_input_width
test_model_input_channels = train_model_input_channels
test_model_pretrained_weights = train_model_pretrained_weights
